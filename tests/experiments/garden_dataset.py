#!/usr/bin/env python3
"""
Garden NP/Z v-trans (mod) dataset.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch as t
from transformers import PreTrainedTokenizer

from task_dataset_base import TaskDatasetBase


def _load_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


@dataclass
class GardenSample:
    clean: str
    corrupted: str
    correct_token: str
    incorrect_token: str
    template: str = ""


class GardenDataset(TaskDatasetBase):
    """
    Exposes:
      - toks / corrupted_toks / input_lengths
      - pos_token_ids / neg_token_ids
      - word_idx with keys: subj, verb, obj_head, rel_pron, rel_verb
      - logit_diff
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[Path] = None,
        samples: Optional[List[GardenSample]] = None,
        prepend_bos: bool = False,
        device: str = "cpu",
        seed: int = 1,
    ):
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self._device = device
        self.seed = seed

        if samples is None:
            if data_path is None:
                raise ValueError("Provide data_path or samples.")
            records = _load_records(data_path)
            samples = self._from_records(records)
        self.samples = samples
        self.N = len(samples)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts_clean = [
            (tokenizer.bos_token if prepend_bos else "") + s.clean for s in samples
        ]
        texts_corr = [
            (tokenizer.bos_token if prepend_bos else "") + s.corrupted for s in samples
        ]

        tok_clean = tokenizer(texts_clean, padding=True, return_attention_mask=True)
        tok_corr = tokenizer(texts_corr, padding=True, return_attention_mask=True)
        self.toks = t.tensor(tok_clean.input_ids).long().to(device)
        self.corrupted_toks = t.tensor(tok_corr.input_ids).long().to(device)
        self.input_lengths = t.tensor(tok_clean.attention_mask).sum(dim=1).long().to(device)

        # Token IDs for logit diff
        self.pos_token_ids = t.tensor(
            [self._token_id(s.correct_token) for s in samples], device=device
        )
        self.neg_token_ids = t.tensor(
            [self._token_id(s.incorrect_token) for s in samples], device=device
        )

        self.word_idx = self._build_word_idx(samples)

    def __len__(self) -> int:
        return self.N

    def _token_id(self, token: str) -> int:
        token = token.strip()
        if token and not token.startswith(" "):
            token = " " + token
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token must be single-token: {token!r} -> {ids}")
        return ids[0]

    def _from_records(self, records: List[Dict[str, Any]]) -> List[GardenSample]:
        samples: List[GardenSample] = []
        for row in records:
            clean = str(row.get("clean", "")).strip()
            corrupted = str(row.get("corrupted", "")).strip()
            correct_token = str(row.get("correct_token", "")).strip()
            incorrect_token = str(row.get("incorrect_token", "")).strip()
            if not clean or not corrupted or not correct_token or not incorrect_token:
                continue
            samples.append(
                GardenSample(
                    clean=clean,
                    corrupted=corrupted,
                    correct_token=correct_token,
                    incorrect_token=incorrect_token,
                    template=str(row.get("template", "")).strip(),
                )
            )
        if not samples:
            raise ValueError("No valid samples found in dataset.")
        return samples

    def _build_word_idx(self, samples: List[GardenSample]) -> List[Dict[str, int]]:
        indices: List[Dict[str, int]] = []
        for s in samples:
            tokens = self.tokenizer.tokenize(s.clean)
            # Expect pattern: SUBORDINATOR the SUBJ VERB the OBJ who REL_VERB ...
            # Use a simple scan to locate key positions.
            subj_idx = verb_idx = obj_idx = rel_pron_idx = rel_verb_idx = None
            for i in range(len(tokens) - 3):
                if tokens[i].lower() in {"as", "while", "after", "when"} and tokens[i + 1] == "Ġthe":
                    subj_idx = i + 2
                    verb_idx = i + 3
                    if i + 4 < len(tokens) and tokens[i + 4] == "Ġthe":
                        obj_idx = i + 5
                    # find relative clause start
                    for j in range(i + 5, len(tokens) - 1):
                        if tokens[j] in {"Ġwho", "who"}:
                            rel_pron_idx = j
                            rel_verb_idx = j + 1
                            break
                    break

            indices.append(
                {
                    "subj": subj_idx if subj_idx is not None else -1,
                    "verb": verb_idx if verb_idx is not None else -1,
                    "obj_head": obj_idx if obj_idx is not None else -1,
                    "rel_pron": rel_pron_idx if rel_pron_idx is not None else -1,
                    "rel_verb": rel_verb_idx if rel_verb_idx is not None else -1,
                }
            )
        return indices

    def logit_diff(self, logits: t.Tensor, mean: bool = True, loss: bool = False) -> t.Tensor:
        idx = t.arange(logits.size(0), device=logits.device)
        pos_logits = logits[idx, self.input_lengths - 1]
        pos = pos_logits.gather(1, self.pos_token_ids.unsqueeze(1)).squeeze(1)
        neg = pos_logits.gather(1, self.neg_token_ids.unsqueeze(1)).squeeze(1)
        results = pos - neg
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    def gen_flipped_prompts(self) -> "GardenDataset":
        flipped = [
            GardenSample(
                clean=s.corrupted,
                corrupted=s.clean,
                correct_token=s.correct_token,
                incorrect_token=s.incorrect_token,
                template=s.template,
            )
            for s in self.samples
        ]
        return GardenDataset(
            tokenizer=self.tokenizer,
            samples=flipped,
            prepend_bos=self.prepend_bos,
            device=self._device,
            seed=self.seed,
        )
