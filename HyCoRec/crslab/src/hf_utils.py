"""Small Hugging Face dataset and chat-template helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_DATASETS_CACHE_DIR = Path(".cache/huggingface/datasets")


def dataset_from_records(records: Sequence[Mapping[str, Any]]) -> Dataset:
    """Create a Dataset, including a valid empty Dataset for empty splits."""

    if records:
        return Dataset.from_list(list(records))
    return Dataset.from_list([])


def dataset_dict_from_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> DatasetDict:
    return DatasetDict({name: dataset_from_records(rows) for name, rows in splits.items()})


def write_dataset_dict_jsonl(dataset_dict: DatasetDict, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, dataset in dataset_dict.items():
        dataset.to_json(str(output_path / f"{split_name}.jsonl"), force_ascii=False)


def load_json_dataset(
    data_files: str | Path | Mapping[str, str | Path],
    cache_dir: str | Path | None = None,
) -> Dataset | DatasetDict:
    cache_path = str(cache_dir or DEFAULT_DATASETS_CACHE_DIR)
    if isinstance(data_files, Mapping):
        normalized = {split: str(path) for split, path in data_files.items()}
        return load_dataset("json", data_files=normalized, cache_dir=cache_path)
    return load_dataset("json", data_files=str(data_files), split="train", cache_dir=cache_path)


def format_messages_with_tokenizer(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    chunks = []
    for message in messages:
        role = str(message.get("role", "user")).title()
        content = message.get("content", "")
        chunks.append(f"{role}: {content}")
    if add_generation_prompt:
        chunks.append("Assistant:")
    return "\n\n".join(chunks)


def sft_text_dataset_from_jsonl(
    data_files: Mapping[str, str | Path],
    tokenizer: Any,
) -> DatasetDict:
    raw_dataset = load_json_dataset(data_files)
    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict when loading SFT train/validation files.")

    def add_text(batch: Mapping[str, Any]) -> dict[str, list[str]]:
        return {
            "text": [
                format_messages_with_tokenizer(messages, tokenizer, add_generation_prompt=False)
                for messages in batch["messages"]
            ]
        }

    remove_columns = {
        split: columns
        for split, columns in raw_dataset.column_names.items()
    }
    formatted_splits = {}
    for split_name, split_dataset in raw_dataset.items():
        formatted_splits[split_name] = split_dataset.map(
            add_text,
            batched=True,
            remove_columns=remove_columns[split_name],
            desc=f"Formatting {split_name} chats",
        )
    return DatasetDict(formatted_splits)
