#!/usr/bin/env python3
"""
Greater-than task dataset utilities.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch as t
from transformers import PreTrainedTokenizer

from task_dataset_base import TaskDatasetBase


def _load_records(path: Path) -> List[Dict[str, str]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as fh:
        return list(csv.DictReader(fh))


@dataclass
class GreaterThanSample:
    clean: str
    corrupted: str
    label: int


class GreaterThanDataset(TaskDatasetBase):
    """
    Dataset wrapper for the classic greater-than task.

    Each sample contains:
      - `clean`: prompt whose hidden threshold is the final two digits of the
        first year, e.g. "... year 1352 to the year 13"
      - `corrupted`: matched prompt with a different threshold, e.g.
        "... year 1301 to the year 13"
      - `label`: the clean threshold, e.g. 52

    The task metric is:
      P(next_two_digit_year > label) - P(next_two_digit_year <= label)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[Path] = None,
        samples: Optional[List[GreaterThanSample]] = None,
        prepend_bos: bool = False,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self._device = device

        if samples is None:
            if data_path is None:
                raise ValueError("Provide data_path or samples.")
            records = _load_records(data_path)
            samples = self._from_records(records)

        self.samples = samples
        self.N = len(samples)
        if self.N == 0:
            raise ValueError("No valid greater-than samples found.")

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
        self.input_lengths = (
            t.tensor(tok_clean.attention_mask).sum(dim=1).long().to(device)
        )

        self.labels = t.tensor([s.label for s in samples], dtype=t.long, device=device)
        self.year_token_ids = t.tensor(
            [self._year_token_id(year) for year in range(100)],
            dtype=t.long,
            device=device,
        )

        # Unused by this task, but kept for compatibility with the shared base.
        self.pos_token_ids = self.year_token_ids
        self.neg_token_ids = self.year_token_ids

    def __len__(self) -> int:
        return self.N

    def _from_records(self, records: List[Dict[str, str]]) -> List[GreaterThanSample]:
        samples: List[GreaterThanSample] = []
        for row in records:
            clean = str(row.get("clean", "")).strip()
            corrupted = str(row.get("corrupted", "")).strip()
            raw_label = str(row.get("label", "")).strip()
            if not clean or not corrupted or not raw_label:
                continue
            samples.append(
                GreaterThanSample(
                    clean=clean,
                    corrupted=corrupted,
                    label=int(raw_label),
                )
            )
        return samples

    def _year_token_id(self, year: int) -> int:
        token = f"{year:02d}"
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Year token must be single-token for GPT-style metric: {token!r} -> {ids}"
            )
        return ids[0]

    def prob_diff(self, logits: t.Tensor, mean: bool = True, loss: bool = False) -> t.Tensor:
        idx = t.arange(logits.size(0), device=logits.device)
        final_logits = logits[idx, self.input_lengths - 1]
        probs = t.softmax(final_logits, dim=-1)[:, self.year_token_ids]

        results = []
        labels = self.labels.to(probs.device)
        for prob, year in zip(probs, labels):
            results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())

        results_tensor = t.stack(results)
        if loss:
            results_tensor = -results_tensor
        if mean:
            results_tensor = results_tensor.mean()
        return results_tensor

    def logit_diff(self, logits: t.Tensor, mean: bool = True, loss: bool = False) -> t.Tensor:
        # Alias for compatibility with shared task interfaces.
        return self.prob_diff(logits, mean=mean, loss=loss)

    def gen_flipped_prompts(self) -> "GreaterThanDataset":
        flipped = [
            GreaterThanSample(
                clean=s.corrupted,
                corrupted=s.clean,
                label=s.label,
            )
            for s in self.samples
        ]
        return GreaterThanDataset(
            tokenizer=self.tokenizer,
            samples=flipped,
            prepend_bos=self.prepend_bos,
            device=self._device,
        )
