"""Run target-conditioned generation and post-check the generated response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .hf_utils import format_messages_with_tokenizer
    from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from .reason_builder import build_evidence, evidence_to_text, post_check_response, render_metadata
except ImportError:  # pragma: no cover
    from HyCoRec.crslab.src.hf_utils import format_messages_with_tokenizer
    from HyCoRec.crslab.src.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from HyCoRec.crslab.src.reason_builder import build_evidence, evidence_to_text, post_check_response, render_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--dialogue_context", required=True)
    parser.add_argument("--recommended_target", required=True)
    parser.add_argument("--target_metadata", default=None, help="JSON string, JSON file path, or plain text metadata.")
    parser.add_argument("--available_evidence", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json_output", action="store_true")
    return parser.parse_args()


def parse_metadata(value: str | None) -> Any:
    if not value:
        return {}
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"description": value}


def build_user_prompt(
    dialogue_context: str,
    recommended_target: str,
    target_metadata: Any,
    available_evidence: str | None,
) -> str:
    evidence_text = available_evidence
    if not evidence_text:
        evidence = build_evidence(dialogue_context, recommended_target, target_metadata)
        evidence_text = evidence_to_text(evidence)
    return USER_PROMPT_TEMPLATE.format(
        dialogue_context=dialogue_context,
        recommended_target=recommended_target,
        target_metadata=render_metadata(target_metadata),
        available_evidence=evidence_text,
    )


def format_prompt(system_prompt: str, user_prompt: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return format_messages_with_tokenizer(messages, tokenizer, add_generation_prompt=True)


def build_generation_config(tokenizer: Any, max_new_tokens: int, temperature: float) -> Any:
    from transformers import GenerationConfig

    config_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        config_kwargs["temperature"] = temperature
    return GenerationConfig(**config_kwargs)


def generate_once(generator: Any, prompt: str, generation_config: Any) -> str:
    outputs = generator(
        prompt,
        generation_config=generation_config,
        return_full_text=False,
    )
    if not outputs:
        return ""
    return outputs[0].get("generated_text", "").strip()


def load_generator(model_name_or_path: str, adapter_path: str | None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )
    return generator, tokenizer


def main() -> None:
    args = parse_args()
    target_metadata = parse_metadata(args.target_metadata)
    generator, tokenizer = load_generator(args.model_name_or_path, args.adapter_path)
    generation_config = build_generation_config(tokenizer, args.max_new_tokens, args.temperature)

    user_prompt = build_user_prompt(
        dialogue_context=args.dialogue_context,
        recommended_target=args.recommended_target,
        target_metadata=target_metadata,
        available_evidence=args.available_evidence,
    )
    prompt = format_prompt(SYSTEM_PROMPT, user_prompt, tokenizer)
    response = generate_once(generator, prompt, generation_config)
    checks = post_check_response(response, args.recommended_target, target_metadata)

    if not checks["target_mentioned"]:
        strict_prompt = user_prompt + (
            "\n\nThe previous response failed to mention the required target item id. "
            f"Regenerate and explicitly mention: {args.recommended_target}."
        )
        response = generate_once(generator, format_prompt(SYSTEM_PROMPT, strict_prompt, tokenizer), generation_config)
        checks = post_check_response(response, args.recommended_target, target_metadata)

    payload = {"response": response, "checks": checks}
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(response)
        print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
