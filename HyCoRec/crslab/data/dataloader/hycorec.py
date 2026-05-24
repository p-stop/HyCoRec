# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import json
import pickle
import random
import re
from copy import deepcopy
from math import ceil
from pathlib import Path

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class HyCoRecDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``"context_truncate"``: the maximum length of context.
        - ``"response_truncate"``: the maximum length of response.
        - ``"entity_truncate"``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``"pad"``
        - ``"start"``
        - ``"end"``
        - ``"pad_entity"``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab["tok2ind"]["__pad__"]
        self.start_token_idx = vocab["tok2ind"]["__start__"]
        self.end_token_idx = vocab["tok2ind"]["__end__"]
        self.split_token_idx = vocab["tok2ind"].get("_split_", None)
        self.related_truncate = opt.get("related_truncate", None)
        self.context_truncate = opt.get("context_truncate", None)
        self.response_truncate = opt.get("response_truncate", None)
        self.entity_truncate = opt.get("entity_truncate", None)
        self.review_entity2id = vocab["entity2id"]
        self.sft_conv_opt = opt['sft_conv']
        self.use_sft_conv = opt.get("use_sft_conv", False)
        self.sft_target_placeholder = str(
            self.sft_conv_opt.get("test_target_placeholder", "<TARGET_ITEMS>")
        )
        work_dir = Path(self.sft_conv_opt.get("work_dir", "./tmp/sft_conv"))
        self.sft_data_dir = Path(self.sft_conv_opt.get("data_dir", work_dir / "data"))
        return

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict["role"] == "Recommender":
                for item in conv_dict["items"]:
                    augment_conv_dict = {
                        "conv_id": conv_dict["conv_id"],
                        "related_item": conv_dict["item"],
                        "related_entity": conv_dict["entity"],
                        "related_word": conv_dict["word"],
                        "item": item,
                    }
                    augment_dataset.append(augment_conv_dict)

        return augment_dataset

    def rec_batchify(self, batch):
        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []
        batch_movies = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_item.append(conv_dict["related_item"])
            batch_related_entity.append(conv_dict["related_entity"])
            batch_related_word.append(conv_dict["related_word"])
            batch_movies.append(conv_dict["item"])
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "conv_id": batch_conv_id,
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "item": torch.tensor(batch_movies, dtype=torch.long),
        }

        return res

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def get_conv_data(self, batch_size, shuffle=True, split=None, force_original=False):
        """Return original CRSLab conv batches or preprocessed SFT batches.

        ``use_sft_conv`` makes the conversation stage read already-built SFT
        records. Recommender-side test prediction can still request the original
        conv batches with ``force_original=True``.
        """

        if self.use_sft_conv and not force_original:
            if split is None:
                raise ValueError("SFT conversation dataloader requires split='train', 'valid', or 'test'.")
            return self.get_sft_data(split=split, batch_size=batch_size, shuffle=shuffle)
        return super().get_conv_data(batch_size=batch_size, shuffle=shuffle)

    def conv_batchify(self, batch):
        batch_related_tokens = []
        batch_context_tokens = []

        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []

        batch_response = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_tokens.append(
                truncate(conv_dict["tokens"][-1], self.related_truncate, truncate_tail=False)
            )
            batch_context_tokens.append(
                truncate(merge_utt(
                    conv_dict["tokens"],
                    start_token_idx=self.start_token_idx,
                    split_token_idx=self.split_token_idx,
                    final_token_idx=self.end_token_idx
                ), self.context_truncate, truncate_tail=False)
            )

            batch_related_item.append(conv_dict["item"])
            batch_related_entity.append(conv_dict["entity"])
            batch_related_word.append(conv_dict["word"])

            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict["response"], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "related_tokens": padded_tensor(batch_related_tokens, self.pad_token_idx, pad_tail=False),
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "response": padded_tensor(batch_response, self.pad_token_idx),
            "conv_id": batch_conv_id,
        }

        return res

    def _normalize_sft_split(self, split):
        split = str(split).lower()
        if split == "validation":
            return "valid"
        return split

    def _sft_file_for_split(self, split):
        split = self._normalize_sft_split(split)
        configured_path = self.sft_conv_opt.get(f"{split}_file")
        if configured_path:
            return Path(configured_path)

        candidates = [
            self.sft_data_dir / f"{split}.jsonl",
            self.sft_data_dir / f"{split}.json",
        ]
        if split == "valid":
            candidates.extend([
                self.sft_data_dir / "validation.jsonl",
                self.sft_data_dir / "validation.json",
            ])
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    def get_sft_file_path(self, split):
        """Public path accessor used by external distributed SFT launchers."""

        return str(self._sft_file_for_split(split))

    def get_sft_records(self, split):
        """Read preprocessed SFT records for one split."""

        path = self._sft_file_for_split(split)
        if not path.exists():
            raise FileNotFoundError(f"SFT {split} file does not exist: {path}")

        if path.suffix.lower() == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as file:
                for line_no, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        else:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            records = self._records_from_json_payload(raw, split)

        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "messages" not in record:
                raise ValueError(f"SFT record {index} in {path} must contain a messages field.")
            normalized.append(record)
        return normalized

    def _records_from_json_payload(self, raw, split):
        split = self._normalize_sft_split(split)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if "messages" in raw:
                return [raw]
            for key in (split, "validation" if split == "valid" else split, "data", "records", "samples"):
                value = raw.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported SFT JSON payload for split {split}.")

    def get_sft_text_dataset(self, split, tokenizer):
        """Return a Hugging Face Dataset with a formatted ``text`` column."""

        from crslab.src.hf_utils import dataset_from_records, format_messages_with_tokenizer

        records = self.get_sft_records(split)
        records = records[:ceil(len(records) * self.scale)]
        dataset = dataset_from_records(records)
        if len(dataset) == 0:
            return dataset

        def add_text(batch):
            return {
                "text": [
                    format_messages_with_tokenizer(messages, tokenizer, add_generation_prompt=False)
                    for messages in batch["messages"]
                ]
            }

        return dataset.map(
            add_text,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Formatting SFT {split} chats",
        )

    def get_sft_data(self, split, batch_size, shuffle=True):
        records = self.get_sft_records(split)
        records = records[:ceil(len(records) * self.scale)]
        batch_num = ceil(len(records) / batch_size) if records else 0
        idx_list = list(range(len(records)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            yield self.sft_batchify([records[idx] for idx in batch_idx])

    def sft_batchify(self, batch):
        metas = [record.get("meta", {}) for record in batch]
        messages = [record["messages"] for record in batch]
        return {
            "messages": messages,
            "meta": metas,
            "target": [meta.get("target", "") for meta in metas],
            "reference": [
                record["messages"][2]["content"] if len(record["messages"]) > 2 else ""
                for record in batch
            ],
            "conv_id": [meta.get("conversation_id", "") for meta in metas],
            "turn_id": [meta.get("turn_id", "") for meta in metas],
        }

    def set_sft_record_target(self, record, target_text):
        """Return a copy of an SFT record whose prompt target has been replaced."""

        updated = deepcopy(record)
        meta = updated.setdefault("meta", {})
        old_target = str(meta.get("target", "") or self.sft_target_placeholder)
        meta.setdefault("original_target", old_target)
        meta["target"] = target_text

        messages = updated.get("messages", [])
        if len(messages) > 1:
            messages[1]["content"] = self._replace_prompt_target(
                messages[1].get("content", ""),
                old_target=old_target,
                target_text=target_text,
            )
        return updated

    def _replace_prompt_target(self, content, old_target, target_text):
        content = str(content)
        for marker in (self.sft_target_placeholder, old_target):
            if marker and marker != target_text and marker in content:
                return content.replace(marker, target_text, 1)

        pattern = re.compile(r"(\[Recommended Target\]\s*)(.*?)(\n\s*\[|$)", re.S)

        def repl(match):
            return f"{match.group(1)}{target_text}{match.group(3)}"

        new_content, count = pattern.subn(repl, content, count=1)
        return new_content if count else content

    def policy_batchify(self, *args, **kwargs):
        pass
