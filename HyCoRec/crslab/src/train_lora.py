"""Fine-tune a target-conditioned generator with Transformers, TRL, and PEFT LoRA."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any

try:
    from .hf_utils import sft_text_dataset_from_jsonl
except ImportError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from crslab.src.hf_utils import sft_text_dataset_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", default="false")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules, or 'auto' to let PEFT infer where supported.",
    )
    return parser.parse_args()


def filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_training_args(args: argparse.Namespace) -> tuple[Any, bool]:
    try:
        from trl import SFTConfig
    except ImportError:
        from transformers import TrainingArguments as SFTConfig

    signature = inspect.signature(SFTConfig.__init__)
    supports_assistant_only = "assistant_only_loss" in signature.parameters
    base_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "dataset_text_field": "text",
        "packing": False,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_seq_length,
        "assistant_only_loss": True,
        "report_to": "none",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "ddp_find_unused_parameters": _str_to_bool(args.ddp_find_unused_parameters),
        "dataloader_num_workers": args.dataloader_num_workers,
        "optim": args.optim,
        "deepspeed": args.deepspeed,
    }
    return SFTConfig(**filtered_kwargs(SFTConfig.__init__, base_kwargs)), supports_assistant_only


def _str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_lora_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType

    target_modules = None
    if args.lora_target_modules.strip().lower() != "auto":
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]

    kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    if target_modules:
        kwargs["target_modules"] = target_modules
    return LoraConfig(**kwargs)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = sft_text_dataset_from_jsonl(
        {"train": args.train_file, "validation": args.valid_file},
        tokenizer,
    )
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    if len(valid_dataset) == 0:
        valid_dataset = train_dataset
    training_args, supports_assistant_only = build_training_args(args)
    if not supports_assistant_only:
        print("Warning: this TRL version does not expose assistant_only_loss; SFTTrainer will train on full chat text.")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "peft_config": build_lora_config(args),
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
    }
    trainer = SFTTrainer(**filtered_kwargs(SFTTrainer.__init__, trainer_kwargs))
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "sft_train_done.txt"), "w", encoding="utf-8") as file:
            file.write("ok\n")


if __name__ == "__main__":
    main()
