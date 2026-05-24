"""Evaluate target-conditioned generation outputs with rule-based safety metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .hf_utils import dataset_from_records, load_json_dataset
    from .reason_builder import post_check_response
except ImportError:  # pragma: no cover
    from HyCoRec.crslab.src.hf_utils import dataset_from_records, load_json_dataset
    from HyCoRec.crslab.src.reason_builder import post_check_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="JSONL/JSON predictions or SFT-style file.")
    parser.add_argument("--output_file", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(text)
        if isinstance(raw, Mapping):
            for key in ("predictions", "data", "records", "samples"):
                if isinstance(raw.get(key), Sequence):
                    return [item for item in raw[key] if isinstance(item, Mapping)]
            return [dict(raw)]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [item for item in raw if isinstance(item, Mapping)]
    except json.JSONDecodeError:
        pass

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc
    return records


def load_records_dataset(path: str | Path) -> Any:
    """Load predictions through datasets.load_dataset, with a permissive fallback."""

    try:
        return load_json_dataset(path)
    except Exception:
        return dataset_from_records(load_records(path))


def extract_response(record: Mapping[str, Any]) -> str:
    for key in ("response", "generated_response", "prediction", "output", "assistant_response"):
        if record.get(key):
            return str(record[key])
    messages = record.get("messages")
    if isinstance(messages, Sequence):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def extract_target(record: Mapping[str, Any]) -> str:
    for key in ("target", "recommended_target", "target_item"):
        if record.get(key):
            return str(record[key])
    meta = record.get("meta")
    if isinstance(meta, Mapping):
        for key in ("target", "recommended_target", "target_item"):
            if meta.get(key):
                return str(meta[key])
    return ""


def extract_reason_strength(record: Mapping[str, Any]) -> str:
    if record.get("reason_strength"):
        return str(record["reason_strength"])
    meta = record.get("meta")
    if isinstance(meta, Mapping) and meta.get("reason_strength"):
        return str(meta["reason_strength"])
    return ""


def extract_metadata(record: Mapping[str, Any]) -> Any:
    for key in ("target_metadata", "metadata", "item_metadata"):
        if key in record:
            return record[key]
    meta = record.get("meta")
    if isinstance(meta, Mapping):
        for key in ("target_metadata", "metadata", "item_metadata"):
            if key in meta:
                return meta[key]
    return {}


def safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    response = extract_response(record)
    target = extract_target(record)
    if not response or not target:
        return {
            "eval_valid": False,
            "target_mentioned_int": 0,
            "target_consistent_int": 0,
            "reason_present_int": 0,
            "weak_evidence_int": 0,
            "weak_evidence_safe_int": 0,
            "hallucination_risk_int": 0,
            "response_length": 0,
        }
    metadata = extract_metadata(record)
    checks = post_check_response(response, target, metadata)
    reason_strength = extract_reason_strength(record)

    weak_evidence = reason_strength == "weak"
    return {
        "eval_valid": True,
        "target_mentioned_int": int(checks["target_mentioned"]),
        "target_consistent_int": int(checks["target_mentioned"] and not checks["off_target_risk"]),
        "reason_present_int": int(checks["reason_present"]),
        "weak_evidence_int": int(weak_evidence),
        "weak_evidence_safe_int": int(weak_evidence and not checks["unsupported_claim_risk"]),
        "hallucination_risk_int": int(checks["unsupported_claim_risk"]),
        "response_length": len(response.split()),
    }


def evaluate_dataset(dataset: Any) -> dict[str, Any]:
    scored = dataset.map(score_record, desc="Scoring generations")
    valid = scored.filter(lambda row: row["eval_valid"], desc="Keeping scorable rows")
    total = len(valid)
    if total == 0:
        return {
            "sample_count": 0,
            "target_mention_rate": 0.0,
            "target_consistency_rate": 0.0,
            "reason_presence_rate": 0.0,
            "weak_evidence_safety_rate": 0.0,
            "average_response_length": 0.0,
            "hallucination_risk_count": 0,
        }

    target_mentions = sum(valid["target_mentioned_int"])
    target_consistent = sum(valid["target_consistent_int"])
    reasons_present = sum(valid["reason_present_int"])
    weak_total = sum(valid["weak_evidence_int"])
    weak_safe = sum(valid["weak_evidence_safe_int"])
    hallucination_risk_count = sum(valid["hallucination_risk_int"])
    response_lengths = valid["response_length"]

    return {
        "sample_count": total,
        "target_mention_rate": safe_rate(target_mentions, total),
        "target_consistency_rate": safe_rate(target_consistent, total),
        "reason_presence_rate": safe_rate(reasons_present, total),
        "weak_evidence_safety_rate": safe_rate(weak_safe, weak_total),
        "average_response_length": round(sum(response_lengths) / len(response_lengths), 2) if response_lengths else 0.0,
        "hallucination_risk_count": hallucination_risk_count,
    }


def evaluate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return evaluate_dataset(dataset_from_records(records))


def main() -> None:
    args = parse_args()
    metrics = evaluate_dataset(load_records_dataset(args.input_file))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
