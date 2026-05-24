"""Evidence, label, and safety helpers for target-conditioned generation."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote

METADATA_FIELDS = (
    "title",
    "genre",
    "director",
    "actors",
    "keywords",
    "description",
    "release_year",
)

USER_ROLES = {"user", "seeker", "human", "customer"}
ASSISTANT_ROLES = {"assistant", "system", "recommender", "bot", "agent"}

PREFERENCE_ALIASES = {
    "science fiction": ("science fiction", "sci-fi", "sci fi", "scifi", "sci-fi"),
    "comedy": ("comedy", "comedies", "funny", "humor", "humour", "lighthearted"),
    "horror": ("horror", "scary", "scare", "frightening", "creepy"),
    "thriller": ("thriller", "suspense", "suspenseful"),
    "action": ("action", "fight", "fights", "explosive"),
    "romance": ("romance", "romantic", "love story"),
    "drama": ("drama", "dramatic"),
    "documentary": ("documentary", "docuseries", "nonfiction", "non-fiction"),
    "animation": ("animation", "animated", "cartoon", "anime"),
    "fantasy": ("fantasy", "magic", "magical"),
    "adventure": ("adventure", "adventurous"),
    "crime": ("crime", "detective", "police", "heist"),
    "mystery": ("mystery", "mysterious"),
    "family": ("family", "kids", "children"),
    "war": ("war", "military"),
    "western": ("western", "cowboy"),
    "musical": ("musical", "music"),
}

REASON_MARKERS = (
    "because",
    "since",
    "as ",
    "fits",
    "fit for",
    "based on",
    "good fit",
    "great",
    "funny",
    "interesting",
    "similar",
    "like",
    "matches",
    "enjoy",
    "you might",
    "you may",
)

UNSUPPORTED_CLAIM_PATTERNS = {
    "director": (r"\bdirected by\b", r"\bdirector\b"),
    "actors": (r"\bstarring\b", r"\bstars\b", r"\bcast\b", r"\bfeaturing\b"),
    "release_year": (
        r"\breleased in\b",
        r"\breleased on\b",
        r"\bcame out in\b",
        r"\bfrom \d{4}\b",
        r"\bstill in theaters\b",
        r"\bcurrently in theaters\b",
        r"\bnew release\b",
    ),
    "awards": (r"\bwon\b", r"\baward", r"\boscar\b", r"\bgolden globe\b"),
    "ratings": (r"\brated\b", r"\brating\b", r"\bimdb\b", r"\brotten tomatoes\b"),
    "description": (r"\btells the story\b", r"\bfollows\b", r"\bis about\b"),
}


@dataclass(frozen=True)
class PreferenceSignal:
    """A user preference found in dialogue text."""

    canonical: str
    alias: str


@dataclass(frozen=True)
class EvidenceResult:
    """Reason strength and evidence for a single SFT sample."""

    reason_strength: str
    evidence: tuple[str, ...]
    matched_preference: str | None = None
    short_supported_reason: str | None = None


def coerce_text(value: Any) -> str:
    """Turn flexible dataset text fields into readable text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_spacing(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return detokenize(value)
    if isinstance(value, Mapping):
        for key in ("text", "utterance", "content", "response"):
            if key in value:
                return coerce_text(value[key])
    return _clean_spacing(str(value))


def detokenize(tokens: Iterable[Any]) -> str:
    text = " ".join(str(token) for token in tokens if token is not None)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+n\s*'\s*t\b", "n't", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+'\s*", "'", text)
    return _clean_spacing(text)


def _clean_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_role(role: Any) -> str:
    raw = str(role or "").strip()
    lowered = raw.lower()
    if lowered in USER_ROLES:
        return "User"
    if lowered in ASSISTANT_ROLES:
        return "Assistant"
    return raw.title() if raw else "Speaker"


def is_assistant_role(role: Any) -> bool:
    return str(role or "").strip().lower() in ASSISTANT_ROLES


def is_user_role(role: Any) -> bool:
    return str(role or "").strip().lower() in USER_ROLES


def clean_item_name(value: Any) -> str:
    """Convert URIs, @ids, and plain values into readable target names."""

    if isinstance(value, Mapping):
        for key in ("title", "name", "item", "movie", "target", "id", "uri"):
            if key in value and value[key]:
                return clean_item_name(value[key])
        return _clean_spacing(str(value))

    text = _clean_spacing(str(value or ""))
    if not text:
        return ""

    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    text = text.strip("\"'")
    text = unquote(text)
    if "/resource/" in text:
        text = text.rsplit("/resource/", 1)[-1]
    elif "/" in text and not text.startswith("@"):
        text = text.rsplit("/", 1)[-1]
    text = text.replace("_", " ")
    text = re.sub(r"\s+\((?:film|.*? film|tv series|television series|franchise)\)$", "", text, flags=re.IGNORECASE)
    return _clean_spacing(text)


def normalize_for_match(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def split_targets(target: str | Sequence[str]) -> list[str]:
    if isinstance(target, Sequence) and not isinstance(target, str):
        return [clean_item_name(item) for item in target if clean_item_name(item)]
    parts = re.split(r"\s*(?:;|\|\|)\s*", str(target or ""))
    return [clean_item_name(part) for part in parts if clean_item_name(part)]


def join_targets(targets: Sequence[str]) -> str:
    return "; ".join(clean_item_name(target) for target in targets if clean_item_name(target))


def natural_join(items: Sequence[str]) -> str:
    cleaned = [clean_item_name(item) for item in items if clean_item_name(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def normalize_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return {str(key): value for key, value in metadata.items() if value not in (None, "", [], {})}
    return {"description": coerce_text(metadata)}


def metadata_to_text(metadata: Any) -> str:
    if metadata is None:
        return ""
    if isinstance(metadata, Mapping):
        pieces = []
        for key, value in metadata.items():
            if value in (None, "", [], {}):
                continue
            pieces.append(str(key))
            pieces.append(metadata_to_text(value))
        return _clean_spacing(" ".join(pieces))
    if isinstance(metadata, Sequence) and not isinstance(metadata, (bytes, bytearray, str)):
        return _clean_spacing(" ".join(metadata_to_text(item) for item in metadata))
    return coerce_text(metadata)


def render_metadata(metadata: Any) -> str:
    metadata = normalize_metadata(metadata)
    if not metadata:
        return "No target metadata provided."

    nested = [key for key, value in metadata.items() if isinstance(value, Mapping)]
    if nested and not any(field in metadata for field in METADATA_FIELDS):
        lines = []
        for target, values in metadata.items():
            if not isinstance(values, Mapping):
                continue
            rendered = _render_flat_metadata(values)
            lines.append(f"{clean_item_name(target)}: {rendered if rendered else 'No metadata provided.'}")
        return "\n".join(lines) if lines else "No target metadata provided."

    rendered = _render_flat_metadata(metadata)
    return rendered if rendered else "No target metadata provided."


def _render_flat_metadata(metadata: Mapping[str, Any]) -> str:
    parts = []
    for field in METADATA_FIELDS:
        if field not in metadata or metadata[field] in (None, "", [], {}):
            continue
        value = metadata[field]
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            value_text = ", ".join(coerce_text(item) for item in value if coerce_text(item))
        else:
            value_text = coerce_text(value)
        if value_text:
            parts.append(f"{field}: {value_text}")
    return "\n".join(parts)


def metadata_has_useful_details(metadata: Any) -> bool:
    text = metadata_to_text(metadata)
    return bool(normalize_for_match(text))


def extract_user_preferences(context_turns: Any) -> list[PreferenceSignal]:
    text = _context_user_text(context_turns)
    lowered = text.lower()
    signals: list[PreferenceSignal] = []
    seen = set()
    for canonical, aliases in PREFERENCE_ALIASES.items():
        for alias in aliases:
            if _contains_phrase(lowered, alias):
                key = (canonical, alias)
                if key not in seen:
                    signals.append(PreferenceSignal(canonical=canonical, alias=alias))
                    seen.add(key)
                break
    return signals


def _context_user_text(context_turns: Any) -> str:
    if isinstance(context_turns, str):
        return context_turns
    pieces = []
    if isinstance(context_turns, Sequence):
        for turn in context_turns:
            if isinstance(turn, Mapping):
                role = turn.get("role") or turn.get("speaker")
                if role and not is_user_role(role):
                    continue
                pieces.append(coerce_text(turn.get("text") or turn.get("utterance") or turn.get("content")))
            else:
                pieces.append(coerce_text(turn))
    return _clean_spacing(" ".join(pieces))


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def build_evidence(context_turns: Any, target: str, metadata: Any = None) -> EvidenceResult:
    """Classify evidence as strong, medium, or weak."""

    metadata = normalize_metadata(metadata)
    metadata_text_norm = normalize_for_match(metadata_to_text(metadata))
    preferences = extract_user_preferences(context_turns)
    evidence: list[str] = []

    for signal in preferences:
        aliases = PREFERENCE_ALIASES.get(signal.canonical, (signal.canonical,))
        if any(_contains_phrase(metadata_text_norm, normalize_for_match(alias)) for alias in aliases):
            evidence.append(f"The user asked for a {signal.canonical} movie.")
            evidence.append(f"The target item has {signal.canonical} metadata.")
            return EvidenceResult(
                reason_strength="strong",
                evidence=tuple(evidence),
                matched_preference=signal.canonical,
                short_supported_reason=f"the target metadata includes {signal.canonical} signals",
            )

    person_match = _find_person_metadata_match(_context_user_text(context_turns), metadata)
    if person_match:
        person_name, field = person_match
        evidence.append(f"The user mentioned {person_name}.")
        evidence.append(f"The target metadata lists {person_name} in {field}.")
        return EvidenceResult(
            reason_strength="strong",
            evidence=tuple(evidence),
            matched_preference=person_name,
            short_supported_reason=f"the target metadata also lists {person_name}",
        )

    if metadata_has_useful_details(metadata):
        genre = _first_metadata_value(metadata, ("genre", "genres"))
        description = _first_metadata_value(metadata, ("description", "plot", "overview"))
        keywords = _first_metadata_value(metadata, ("keywords", "tags"))
        if preferences:
            evidence.append(f"The user expressed interest in {preferences[0].canonical}.")
        if genre:
            evidence.append(f"The target item belongs to the {genre} genre.")
        elif keywords:
            evidence.append(f"The target metadata includes keywords: {keywords}.")
        elif description:
            evidence.append("The target metadata includes a description.")
        else:
            evidence.append("Some target metadata is available.")
        return EvidenceResult(
            reason_strength="medium",
            evidence=tuple(evidence),
            short_supported_reason="the available target details provide a cautious basis for the recommendation",
        )

    return EvidenceResult(
        reason_strength="weak",
        evidence=("Evidence is insufficient for a specific personalized reason.",),
    )


def _find_person_metadata_match(user_text: str, metadata: Mapping[str, Any]) -> tuple[str, str] | None:
    user_norm = normalize_for_match(user_text)
    for field in ("director", "actors"):
        names = _metadata_values(metadata, field)
        for name in names:
            name_text = coerce_text(name)
            name_norm = normalize_for_match(name_text)
            if len(name_norm) >= 4 and name_norm in user_norm:
                return name_text, field
    for value in metadata.values():
        if isinstance(value, Mapping):
            match = _find_person_metadata_match(user_text, value)
            if match:
                return match
    return None


def _metadata_values(metadata: Mapping[str, Any], field: str) -> list[Any]:
    if field not in metadata:
        return []
    value = metadata[field]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return list(value)
    return [value]


def _first_metadata_value(metadata: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        if field in metadata and metadata[field] not in (None, "", [], {}):
            value = metadata[field]
            if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
                return ", ".join(coerce_text(item) for item in value if coerce_text(item))
            return coerce_text(value)
    for value in metadata.values():
        if isinstance(value, Mapping):
            nested = _first_metadata_value(value, fields)
            if nested:
                return nested
    return ""


def evidence_to_text(evidence: EvidenceResult | Sequence[str]) -> str:
    if isinstance(evidence, EvidenceResult):
        items = evidence.evidence
    else:
        items = tuple(evidence)
    return "\n".join(f"- {item}" for item in items) if items else "- Evidence is insufficient for a specific personalized reason."


def replace_item_placeholders(raw_response: str, targets: Sequence[str]) -> str:
    """Replace @movie ids in response with target names by mention order."""

    response = coerce_text(raw_response)
    mentions = list(dict.fromkeys(re.findall(r"@\d+", response)))
    cleaned_targets = [clean_item_name(target) for target in targets if clean_item_name(target)]
    for mention, target in zip(mentions, cleaned_targets):
        response = response.replace(mention, target)
    return _clean_spacing(response)


def response_has_raw_reason(raw_response: str, target: str | Sequence[str]) -> bool:
    response = coerce_text(raw_response)
    lowered = response.lower()
    if any(marker in lowered for marker in REASON_MARKERS):
        return True

    reduced = response
    for item in split_targets(target):
        reduced = re.sub(re.escape(item), " ", reduced, flags=re.IGNORECASE)
    reduced = re.sub(r"@\d+", " ", reduced)
    reduced = re.sub(r"\b(?:you|should|watch|see|try|recommend|i|would|d|movie|film|yes|no|it|is|a|an|the)\b", " ", reduced, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z0-9]+", reduced)
    return len(tokens) >= 4


def build_enhanced_response(
    target: str | Sequence[str],
    metadata: Any,
    evidence: EvidenceResult,
    raw_response: str | None = None,
) -> str:
    targets = split_targets(target)
    target_text = natural_join(targets)
    if raw_response:
        response = _clean_spacing(raw_response)
        if not unsupported_claims(response, metadata):
            if not target_mentioned(response, target):
                response = f"I'd recommend {target_text}. {response}"
            limited = _limit_to_two_sentences(response)
            return limited if target_mentioned(limited, target) else response

    if evidence.reason_strength == "strong":
        matched = evidence.matched_preference or "your stated preferences"
        supported = evidence.short_supported_reason or "the available evidence supports the match"
        return f"I'd recommend {target_text}. It fits your interest in {matched}, and {supported}."
    if evidence.reason_strength == "medium":
        return f"I'd recommend {target_text}. Based on its available details, it seems like a good fit for what you're asking for."
    return f"I'd recommend {target_text}. It's a reasonable starting point, and I can narrow it down further if you tell me what mood or genre you prefer."


def _limit_to_two_sentences(response: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", response)
    if len(parts) <= 2:
        return response
    return _clean_spacing(" ".join(parts[:2]))


def target_mentioned(response: str, target: str | Sequence[str]) -> bool:
    response_norm = normalize_for_match(response)
    targets = split_targets(target)
    if not targets:
        return False

    for item in targets:
        item_norm = normalize_for_match(item)
        if not item_norm:
            return False
        if item_norm in response_norm:
            continue
        if _fuzzy_contains(response_norm, item_norm):
            continue
        return False
    return True


def _fuzzy_contains(response_norm: str, item_norm: str) -> bool:
    response_tokens = response_norm.split()
    item_tokens = item_norm.split()
    if not response_tokens or not item_tokens:
        return False
    window = max(1, len(item_tokens))
    for i in range(max(1, len(response_tokens) - window + 1)):
        candidate = " ".join(response_tokens[i : i + window])
        if difflib.SequenceMatcher(None, candidate, item_norm).ratio() >= 0.88:
            return True
    return difflib.SequenceMatcher(None, response_norm, item_norm).ratio() >= 0.75


def reason_present(response: str, target: str | Sequence[str]) -> bool:
    lowered = response.lower()
    if any(marker in lowered for marker in REASON_MARKERS):
        return True

    reduced = response
    for item in split_targets(target):
        reduced = re.sub(re.escape(item), " ", reduced, flags=re.IGNORECASE)
    reduced = re.sub(r"\b(?:i|would|d|recommend|suggest|watch|see|try|you|should|the|a|an|movie|film)\b", " ", reduced, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z0-9]+", reduced)
    return len(tokens) >= 5


def off_target_risk(response: str, target: str | Sequence[str]) -> bool:
    """Conservative rule for extra recommended titles."""

    response_norm = normalize_for_match(response)
    target_norms = [normalize_for_match(item) for item in split_targets(target)]
    risky_markers = ("also recommend", "another recommendation", "you could also", "i'd also", "i would also")
    if not any(marker in response_norm for marker in risky_markers):
        return False

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', response)
    candidates = [a or b for a, b in quoted]
    candidates.extend(re.findall(r"\b(?:[A-Z][A-Za-z0-9:&'\-]+(?:\s+|$)){2,}", response))
    for candidate in candidates:
        candidate_norm = normalize_for_match(candidate)
        if not candidate_norm or candidate_norm in {"based on", "the target", "i would"}:
            continue
        if not any(candidate_norm in target_norm or target_norm in candidate_norm for target_norm in target_norms):
            return True
    return False


def unsupported_claims(response: str, metadata: Any = None) -> list[str]:
    metadata = normalize_metadata(metadata)
    response_norm = response.lower()
    risks = []
    for field, patterns in UNSUPPORTED_CLAIM_PATTERNS.items():
        if not any(re.search(pattern, response_norm) for pattern in patterns):
            continue
        if not _metadata_supports_claim(response, metadata, field):
            risks.append(field)
    return risks


def _metadata_supports_claim(response: str, metadata: Mapping[str, Any], field: str) -> bool:
    if field in {"awards", "ratings"}:
        return field in metadata and bool(metadata[field])
    if field == "release_year":
        years = re.findall(r"\b\d{4}\b", response)
        metadata_years = re.findall(r"\b\d{4}\b", metadata_to_text(metadata))
        return bool(years) and all(year in metadata_years for year in years)
    values = _metadata_values(metadata, field)
    if not values and field == "description":
        values = _metadata_values(metadata, "plot") + _metadata_values(metadata, "overview")
    if not values:
        for value in metadata.values():
            if isinstance(value, Mapping) and _metadata_supports_claim(response, value, field):
                return True
        return False
    response_norm = normalize_for_match(response)
    return any(normalize_for_match(value) in response_norm for value in values if normalize_for_match(value))


def post_check_response(response: str, target: str | Sequence[str], metadata: Any = None) -> dict[str, Any]:
    claims = unsupported_claims(response, metadata)
    return {
        "target_mentioned": target_mentioned(response, target),
        "off_target_risk": off_target_risk(response, target),
        "reason_present": reason_present(response, target),
        "unsupported_claim_risk": bool(claims),
        "unsupported_claims": claims,
    }
