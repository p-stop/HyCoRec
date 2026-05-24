"""Build target-conditioned SFT JSONL data from flexible CRS datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .hf_utils import dataset_dict_from_splits, write_dataset_dict_jsonl
    from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from .reason_builder import (
        build_enhanced_response,
        build_evidence,
        clean_item_name,
        coerce_text,
        evidence_to_text,
        is_assistant_role,
        join_targets,
        metadata_has_useful_details,
        normalize_for_match,
        normalize_metadata,
        normalize_role,
        render_metadata,
        replace_item_placeholders,
        response_has_raw_reason,
    )
except ImportError:  # pragma: no cover - allows running as python src/build_sft_dataset.py
    from hf_utils import dataset_dict_from_splits, write_dataset_dict_jsonl
    from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    from reason_builder import (
        build_enhanced_response,
        build_evidence,
        clean_item_name,
        coerce_text,
        evidence_to_text,
        is_assistant_role,
        join_targets,
        metadata_has_useful_details,
        normalize_for_match,
        normalize_metadata,
        normalize_role,
        render_metadata,
        replace_item_placeholders,
        response_has_raw_reason,
    )


CONVERSATION_KEYS = ("dialog", "dialogue", "conversation", "conversations", "turns", "messages")
CONVERSATION_ID_KEYS = ("conv_id", "conversation_id", "dialog_id", "id")
TURN_ID_KEYS = ("utt_id", "turn_id", "id")
ROLE_KEYS = ("role", "speaker", "sender", "from")
TEXT_KEYS = ("text", "utterance", "content", "response", "message")
TARGET_KEYS = (
    "recommended_target",
    "target_item",
    "target_items",
    "target",
    "targets",
    "recommendation",
    "recommendations",
    "movie",
    "movies",
    "item",
    "items",
)
METADATA_KEY_FIELDS = ("id", "item_id", "movie_id", "dbpedia_id", "dbpedia_uri", "uri", "url", "title", "name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", default=None, help="Raw JSON/JSONL conversation file.")
    parser.add_argument("--train_file", default=None, help="Optional raw train JSON/JSONL conversation file.")
    parser.add_argument("--valid_file", default=None, help="Optional raw valid JSON/JSONL conversation file.")
    parser.add_argument("--test_file", default=None, help="Optional raw test JSON/JSONL conversation file.")
    parser.add_argument("--metadata_file", default=None, help="Optional item metadata JSON/JSONL file.")
    parser.add_argument("--output_dir", default="data/processed", help="Directory for train/valid/test JSONL files.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--blank_test_targets",
        action="store_true",
        help="Replace test prompt targets with a placeholder to be filled by the recommender at test time.",
    )
    parser.add_argument(
        "--test_target_placeholder",
        default="<TARGET_ITEMS>",
        help="Placeholder inserted into test prompts when --blank_test_targets is used.",
    )
    parser.add_argument(
        "--save_hf_dataset",
        action="store_true",
        help="Also save the generated DatasetDict with Hugging Face save_to_disk().",
    )
    return parser.parse_args()


def load_json_flexible(path: str | Path) -> Any:
    """Read JSON, JSON with trailing commas, or JSONL."""

    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    for candidate in (text, _remove_trailing_commas(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(_remove_trailing_commas(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSONL line {line_no} in {path}: {exc}") from exc
    return records


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def iter_conversations(raw: Any) -> list[Mapping[str, Any]]:
    """Flatten common raw dataset wrappers into conversation dictionaries."""

    if isinstance(raw, Mapping):
        for key in ("train", "valid", "validation", "test"):
            if key in raw:
                conversations = []
                for split_key in ("train", "valid", "validation", "test"):
                    if split_key in raw:
                        conversations.extend(iter_conversations(raw[split_key]))
                return conversations
        for key in ("conversations", "dialogs", "dialogues", "data", "records", "samples"):
            if key in raw:
                return iter_conversations(raw[key])
        if any(key in raw for key in CONVERSATION_KEYS):
            return [raw]
        return [raw]

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        conversations = []
        for item in raw:
            conversations.extend(iter_conversations(item))
        return conversations
    return []


def get_turns(conversation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in CONVERSATION_KEYS:
        value = conversation.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [turn for turn in value if isinstance(turn, Mapping)]
    return []


def get_first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def extract_targets(turn: Mapping[str, Any]) -> list[str]:
    text = get_first(turn, TEXT_KEYS, "")
    text_value = coerce_text(text)
    item_ids = _extract_item_ids(text_value)
    if item_ids:
        return item_ids

    candidates: list[Any] = []
    for key in TARGET_KEYS:
        if key in turn:
            candidates.extend(_flatten_item_values(turn[key]))

    if not candidates:
        candidates.extend(item_ids)

    cleaned: list[str] = []
    seen = set()
    for candidate in candidates:
        name = clean_item_name(candidate)
        if not name:
            continue
        key = normalize_for_match(name)
        if key and key not in seen:
            cleaned.append(name)
            seen.add(key)
    return cleaned


def _extract_item_ids(text: str) -> list[str]:
    item_ids: list[str] = []
    seen = set()
    for item_id in re.findall(r"@\d+", text):
        if item_id not in seen:
            item_ids.append(item_id)
            seen.add(item_id)
    return item_ids


def _flatten_item_values(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, Mapping):
        if any(key in value for key in METADATA_KEY_FIELDS):
            return [value]
        flattened: list[Any] = []
        for key in TARGET_KEYS + METADATA_KEY_FIELDS:
            if key in value:
                flattened.extend(_flatten_item_values(value[key]))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_item_values(item))
        return flattened
    return [value]


def build_metadata_index(path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    if not path:
        return {}
    raw = load_json_flexible(path)
    index: dict[str, Mapping[str, Any]] = {}

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                _add_metadata_record(index, value, extra_keys=[key])
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    if isinstance(item, Mapping):
                        _add_metadata_record(index, item, extra_keys=[key])
    else:
        for record in iter_conversations(raw):
            _add_metadata_record(index, record)
    return index


def _add_metadata_record(
    index: dict[str, Mapping[str, Any]],
    record: Mapping[str, Any],
    extra_keys: Sequence[Any] | None = None,
) -> None:
    metadata = normalize_metadata(record)
    keys = list(extra_keys or [])
    for field in METADATA_KEY_FIELDS:
        if field in record:
            keys.extend(_flatten_item_values(record[field]))
    for key in keys:
        for variant in (str(key), clean_item_name(key)):
            norm = normalize_for_match(variant)
            if norm:
                index[norm] = metadata


def metadata_for_targets(targets: Sequence[str], metadata_index: Mapping[str, Mapping[str, Any]]) -> Any:
    matched = {}
    for target in targets:
        metadata = metadata_index.get(normalize_for_match(target))
        if metadata:
            matched[target] = metadata
    if len(targets) == 1:
        return matched.get(targets[0], {})
    return matched


def format_dialogue_context(context_turns: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for turn in context_turns:
        role = normalize_role(get_first(turn, ROLE_KEYS, "Speaker"))
        text = coerce_text(get_first(turn, TEXT_KEYS, ""))
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) if lines else "No prior dialogue context."


def build_samples(
    conversations: Sequence[Mapping[str, Any]],
    metadata_index: Mapping[str, Mapping[str, Any]],
    *,
    allow_missing_targets: bool = False,
    missing_target_placeholder: str = "<TARGET_ITEMS>",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    stats = {
        "conversation_ids": set(),
        "recommender_turn_count": 0,
        "target_turn_count": 0,
        "metadata_count": 0,
        "raw_reason_count": 0,
        "reason_strength": Counter(),
        "context_lengths": [],
        "response_lengths": [],
    }

    for conv_index, conversation in enumerate(conversations):
        conv_id = str(get_first(conversation, CONVERSATION_ID_KEYS, f"conversation_{conv_index}"))
        stats["conversation_ids"].add(conv_id)
        turns = get_turns(conversation)

        for turn_index, turn in enumerate(turns):
            role = get_first(turn, ROLE_KEYS, "")
            if not is_assistant_role(role):
                continue
            stats["recommender_turn_count"] += 1

            targets = extract_targets(turn)
            missing_target = False
            if not targets:
                if not allow_missing_targets:
                    continue
                missing_target = True

            if missing_target:
                target_text = missing_target_placeholder
            else:
                stats["target_turn_count"] += 1
                target_text = join_targets(targets)

            turn_id = str(get_first(turn, TURN_ID_KEYS, turn_index))
            context_turns = turns[:turn_index]
            dialogue_context = format_dialogue_context(context_turns)
            metadata = {} if missing_target else metadata_for_targets(targets, metadata_index)
            evidence = build_evidence(context_turns, target_text, metadata)
            raw_text = coerce_text(get_first(turn, TEXT_KEYS, ""))
            cleaned_raw = raw_text if missing_target else replace_item_placeholders(raw_text, targets)
            has_raw_reason = False if missing_target else response_has_raw_reason(cleaned_raw, target_text)
            if missing_target and cleaned_raw:
                response = cleaned_raw
            else:
                response = build_enhanced_response(
                    target=target_text,
                    metadata=metadata,
                    evidence=evidence,
                    raw_response=cleaned_raw if has_raw_reason else None,
                )
            user_prompt = USER_PROMPT_TEMPLATE.format(
                dialogue_context=dialogue_context,
                recommended_target=target_text,
                target_metadata=render_metadata(metadata),
                available_evidence=evidence_to_text(evidence),
            )
            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                ],
                "meta": {
                    "conversation_id": conv_id,
                    "turn_id": turn_id,
                    "target": target_text,
                    "reason_strength": evidence.reason_strength,
                    "has_raw_reason": has_raw_reason,
                    "target_missing": missing_target,
                },
            }
            if missing_target:
                sample["meta"]["original_target"] = ""
            samples.append(sample)

            if metadata_has_useful_details(metadata):
                stats["metadata_count"] += 1
            if has_raw_reason:
                stats["raw_reason_count"] += 1
            stats["reason_strength"][evidence.reason_strength] += 1
            stats["context_lengths"].append(len(dialogue_context.split()))
            stats["response_lengths"].append(len(response.split()))

    return samples, stats


def split_records_by_conversation(
    samples: Sequence[dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio < 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be non-negative and sum to less than 1.")

    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_conv[sample["meta"]["conversation_id"]].append(sample)

    conv_ids = list(by_conv)
    random.Random(seed).shuffle(conv_ids)

    n_total = len(conv_ids)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    if n_total and n_train == 0 and train_ratio > 0:
        n_train = 1
    train_ids = set(conv_ids[:n_train])
    valid_ids = set(conv_ids[n_train : n_train + n_valid])

    splits = {"train": [], "valid": [], "test": []}
    for conv_id in conv_ids:
        if conv_id in train_ids:
            split = "train"
        elif conv_id in valid_ids:
            split = "valid"
        else:
            split = "test"
        splits[split].extend(by_conv[conv_id])
    return splits


def split_by_conversation(
    samples: Sequence[dict[str, Any]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
):
    """Return a Hugging Face DatasetDict split by conversation id."""

    split_records = split_records_by_conversation(samples, train_ratio, valid_ratio, seed)
    return dataset_dict_from_splits(split_records)


def replace_test_targets_with_placeholder(
    samples: Sequence[dict[str, Any]],
    placeholder: str,
) -> list[dict[str, Any]]:
    """Blank test prompt targets while keeping the assistant reference intact."""

    blanked = []
    for sample in samples:
        updated = json.loads(json.dumps(sample, ensure_ascii=False))
        meta = updated.setdefault("meta", {})
        original_target = str(meta.get("original_target", "") if meta.get("target_missing") else meta.get("target", "") or "")
        meta["original_target"] = original_target
        meta["target"] = placeholder

        messages = updated.get("messages", [])
        if len(messages) > 1:
            messages[1]["content"] = replace_recommended_target_in_prompt(
                messages[1].get("content", ""),
                old_target=original_target,
                new_target=placeholder,
            )
        blanked.append(updated)
    return blanked


def replace_recommended_target_in_prompt(content: str, old_target: str, new_target: str) -> str:
    content = str(content)
    if old_target and old_target in content:
        return content.replace(old_target, new_target, 1)

    pattern = re.compile(r"(\[Recommended Target\]\s*)(.*?)(\n\s*\[|$)", re.S)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{new_target}{match.group(3)}"

    replaced, count = pattern.subn(repl, content, count=1)
    return replaced if count else content


def build_samples_from_file(
    path: str | Path,
    metadata_index: Mapping[str, Mapping[str, Any]],
    *,
    allow_missing_targets: bool = False,
    missing_target_placeholder: str = "<TARGET_ITEMS>",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = load_json_flexible(path)
    conversations = iter_conversations(raw)
    return build_samples(
        conversations,
        metadata_index,
        allow_missing_targets=allow_missing_targets,
        missing_target_placeholder=missing_target_placeholder,
    )


def merge_stats(stats_list: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = {
        "conversation_ids": set(),
        "recommender_turn_count": 0,
        "target_turn_count": 0,
        "metadata_count": 0,
        "raw_reason_count": 0,
        "reason_strength": Counter(),
        "context_lengths": [],
        "response_lengths": [],
    }
    for stats in stats_list:
        merged["conversation_ids"].update(stats["conversation_ids"])
        for key in ("recommender_turn_count", "target_turn_count", "metadata_count", "raw_reason_count"):
            merged[key] += stats[key]
        merged["reason_strength"].update(stats["reason_strength"])
        merged["context_lengths"].extend(stats["context_lengths"])
        merged["response_lengths"].extend(stats["response_lengths"])
    return merged


def summarize_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    sample_count = sum(stats["reason_strength"].values())
    recommender_turns = stats["recommender_turn_count"]
    context_lengths = stats["context_lengths"]
    response_lengths = stats["response_lengths"]
    return {
        "number_of_conversations": len(stats["conversation_ids"]),
        "number_of_recommendation_samples": sample_count,
        "recommender_turns_seen": recommender_turns,
        "target_coverage": _safe_ratio(stats["target_turn_count"], recommender_turns),
        "metadata_coverage": _safe_ratio(stats["metadata_count"], sample_count),
        "reason_strength_distribution": dict(stats["reason_strength"]),
        "raw_response_reason_percentage": _safe_ratio(stats["raw_reason_count"], sample_count),
        "average_dialogue_context_length": _average(context_lengths),
        "average_response_length": _average(response_lengths),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _average(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def main() -> None:
    args = parse_args()
    metadata_index = build_metadata_index(args.metadata_file)

    split_files = {
        "train": args.train_file,
        "valid": args.valid_file,
        "test": args.test_file,
    }
    provided_split_files = {split: path for split, path in split_files.items() if path}
    if provided_split_files:
        missing = [split for split, path in split_files.items() if not path]
        if missing:
            raise ValueError(f"Split-file mode requires train_file, valid_file, and test_file. Missing: {missing}")

        split_records = {}
        split_stats = []
        for split, path in split_files.items():
            allow_missing_targets = split == "test" and args.blank_test_targets
            samples, stats = build_samples_from_file(
                path,
                metadata_index,
                allow_missing_targets=allow_missing_targets,
                missing_target_placeholder=args.test_target_placeholder,
            )
            if split == "test" and args.blank_test_targets:
                samples = replace_test_targets_with_placeholder(samples, args.test_target_placeholder)
            split_records[split] = samples
            split_stats.append(stats)
        stats = merge_stats(split_stats)
        splits = dataset_dict_from_splits(split_records)
    else:
        if not args.input_file:
            raise ValueError("Provide either --input_file or all of --train_file/--valid_file/--test_file.")

        raw = load_json_flexible(args.input_file)
        conversations = iter_conversations(raw)
        samples, stats = build_samples(conversations, metadata_index)
        split_records = split_records_by_conversation(samples, args.train_ratio, args.valid_ratio, args.seed)
        if args.blank_test_targets:
            split_records["test"] = replace_test_targets_with_placeholder(
                split_records["test"],
                args.test_target_placeholder,
            )
        splits = dataset_dict_from_splits(split_records)

    output_dir = Path(args.output_dir)
    write_dataset_dict_jsonl(splits, output_dir)
    if args.save_hf_dataset:
        splits.save_to_disk(str(output_dir / "hf_dataset"))

    summary = summarize_stats(stats)
    summary["split_sizes"] = {name: len(dataset) for name, dataset in splits.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
