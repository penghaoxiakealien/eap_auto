#!/usr/bin/env python3
"""
AgrGenderDataset
---------------
仿 IOIDataset 接口的性别一致性数据类，继承 TaskDatasetBase：
- 前缀文本（不含代词），在末尾预测 he/she
- label: 0 -> 期望 he；1 -> 期望 she
- 暴露 toks/corrupted_toks/input_lengths/pos_token_ids/neg_token_ids，提供 logit_diff
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch as t
from transformers import PreTrainedTokenizer

from task_dataset_base import TaskDatasetBase

# 默认名字/动词/连词
MALE_NAMES = [
    "Gary", "Daniel", "Tom", "Michael", "Peter", "John", "Paul", "Robert", "Tony", "Henry",
    "Alan", "Jake", "Luke", "Chris", "David", "George", "Brian", "Edward", "Frank", "Kevin",
]
FEMALE_NAMES = [
    "Laura", "Linda", "Amy", "Rose", "Lisa", "Mary", "Anna", "Emma", "Julia", "Kelly",
    "Alice", "Sarah", "Rachel", "Diana", "Sophie", "Emily", "Chloe", "Grace", "Hannah", "Karen",
]

VERBS = [
    "ran", "walked", "left", "disappeared", "hurried", "laughed", "cried", "traveled", "studied",
    "went home", "went to school", "read books", "took a taxi", "played piano", "bought food",
]
CONJ = ["because", "after", "while", "when", "although"]

# 两句式（句末固定 because，预测 he/she）模板组件
TWO_SENTENCE_EVENTS = [
    "argued all day",
    "worked together",
    "missed the train",
    "talked for hours",
    "waited outside",
    "studied all night",
]
TWO_SENTENCE_INTROS = [
    "In the evening,",
    "Later,",
    "Afterward,",
    "In the end,",
    "That night,",
]
TWO_SENTENCE_VERBS = [
    "apologized",
    "explained",
    "complained",
    "agreed",
    "insisted",
    "responded",
]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        records: List[Dict[str, Any]] = []
        with path.open() as f:
            if path.suffix.lower() == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    raise ValueError("JSON must be a list of records.")
        return records
    records = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


@dataclass
class AgrGenderSample:
    clean: str
    corrupted: str
    label: int  # 0 -> he, 1 -> she
    S: str      # 主语名字
    D: str      # 干扰名字（与 S 性别相反）
    PR: str     # 目标代词（he/she）
    verb: str   # 动作（可能是短语）
    template: str = ""
    verb_anchor: str = ""  # 动词或短语首词，用于定位


class AgrGenderDataset(TaskDatasetBase):
    """
    关键暴露：
      - toks / corrupted_toks / input_lengths
      - pos_token_ids（he/she 按 label）/ neg_token_ids（相反）
      - word_idx: [{"end": idx}] 预测位置为末位
      - logit_diff: 在末位计算 he/she 概率差
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: Optional[Path] = None,
        samples: Optional[List[AgrGenderSample]] = None,
        prepend_bos: bool = False,
        device: str = "cpu",
        seed: int = 1,
        generate_N: Optional[int] = None,
    ):
        random.seed(seed)
        # 在生成样本前就绑定 tokenizer，便于过滤单 token 名字，保证 clean/corrupted 长度一致
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self._device = device
        if samples is None:
            if data_path is not None:
                records = _load_records(data_path)
                samples = self._from_records(records)
            elif generate_N is not None:
                samples = self._generate_samples(generate_N)
            else:
                raise ValueError("Provide data_path or samples or generate_N")
        self.samples = samples
        self.N = len(samples)
        self.seed = seed

        self.prompts = []
        for idx, s in enumerate(samples):
            self.prompts.append(
                {
                    "text": s.clean,
                    "S": s.S,
                    "D": s.D,
                    "PR": s.PR,
                    "VERB": s.verb,
                    "TEMPLATE_IDX": idx,
                }
            )

        self.sentences = [p["text"] for p in self.prompts]
        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts_clean = [
            (tokenizer.bos_token if prepend_bos else "") + p["text"] for p in self.prompts
        ]
        texts_corr = [
            (tokenizer.bos_token if prepend_bos else "") + s.corrupted for s in samples
        ]

        tok_clean = tokenizer(texts_clean, padding=True, return_attention_mask=True)
        tok_corr = tokenizer(texts_corr, padding=True, return_attention_mask=True)
        self.toks = t.tensor(tok_clean.input_ids).long().to(device)
        self.corrupted_toks = t.tensor(tok_corr.input_ids).long().to(device)
        true_lengths = t.tensor(tok_clean.attention_mask).sum(dim=1).long()
        self.input_lengths = true_lengths.to(device)

        # token IDs
        self.he_token_id = tokenizer(" he", add_special_tokens=False).input_ids[0]
        self.she_token_id = tokenizer(" she", add_special_tokens=False).input_ids[0]
        self.pos_token_ids = t.tensor(
            [self.he_token_id if s.PR.lower() == "he" else self.she_token_id for s in samples],
            device=device,
        )
        self.neg_token_ids = t.tensor(
            [self.she_token_id if s.PR.lower() == "he" else self.he_token_id for s in samples],
            device=device,
        )
        self.s_tokenIDs = t.tensor([tokenizer.encode(" " + s.S)[0] for s in samples]).to(device)

        # 记录 END / VERB / A1 / B / A2 等关键位置
        # - END：最后一个输入 token（用于取 logits 预测下一 token）
        # - VERB：第二句动词 token（默认用动词首词定位）
        # - A1：第一句第一个名字（句首）
        # - B：第一句第二个名字（"and" 后的名字）
        # - A2：第二句再次出现的 A（通常在逗号后）
        self.word_idx = []

        def find_subseq(tokens: List[str], pattern: List[str], start: int = 0, end: Optional[int] = None) -> Optional[int]:
            if not pattern:
                return None
            if end is None:
                end = len(tokens)
            last = end - len(pattern)
            for i0 in range(start, last + 1):
                if tokens[i0 : i0 + len(pattern)] == pattern:
                    return i0
            return None

        for i, s in enumerate(samples):
            tokens = tokenizer.convert_ids_to_tokens(self.toks[i].tolist())
            end_pos = int(true_lengths[i].item() - 1)
            tokens_trimmed = tokens[: end_pos + 1]

            # A1：句首名字（不加前导空格）
            a1_pat = tokenizer.tokenize((s.S or "").strip())
            a1_pos = find_subseq(tokens_trimmed, a1_pat, start=0)

            # B：and 后的名字（加前导空格）
            b_pat = tokenizer.tokenize(" " + (s.D or "").strip())
            b_pos = find_subseq(tokens_trimmed, b_pat, start=0)

            # A2：第二次出现的名字（加前导空格），从 B 之后开始找
            a2_pat = tokenizer.tokenize(" " + (s.S or "").strip())
            start_after = (b_pos + 1) if isinstance(b_pos, int) else 0
            a2_pos = find_subseq(tokens_trimmed, a2_pat, start=start_after)

            # VERB：动词首词（加前导空格）
            verb_anchor = (s.verb_anchor or s.verb or "").strip()
            verb_pat = tokenizer.tokenize(" " + verb_anchor) if verb_anchor else []
            verb_pos = find_subseq(tokens_trimmed, verb_pat, start=(a2_pos or 0))

            self.word_idx.append(
                {
                    "end": end_pos,
                    "verb": verb_pos,
                    "a1": a1_pos,
                    "b": b_pos,
                    "a2": a2_pos,
                }
            )

        self.max_len = self.toks.size(1)
        self.tokenized_prompts = [
            "|".join([tokenizer.decode(tok) for tok in row]) for row in self.toks
        ]

    # ---- 生成/解析辅助 ----
    def _single_token_names(self, names: List[str]) -> List[str]:
        """只保留在 GPT-2 BPE 下（带前导空格）能被编码为单 token 的名字。"""
        good = []
        for name in names:
            ids = self.tokenizer(" " + name, add_special_tokens=False).input_ids
            if isinstance(ids, list) and len(ids) == 1:
                good.append(name)
        return good

    def _generate_samples(self, N: int) -> List[AgrGenderSample]:
        # 为了兼容 EAP-IG：同一个 batch 内 clean/corrupted 的 token 长度必须一致。
        # 这里直接用 tokenizer 对 clean/corrupted 编码，若长度不一致则丢弃重采样。
        male_names = list(MALE_NAMES)
        female_names = list(FEMALE_NAMES)
        if not male_names or not female_names:
            raise ValueError("名字列表为空。")

        res = []
        attempts = 0
        max_attempts = max(1000, N * 200)
        while len(res) < N:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"生成样本失败：尝试 {attempts} 次仍无法凑够 {N} 条等长 (clean/corrupted) 样本。"
                )
            if random.random() < 0.5:
                name = random.choice(male_names)
                pr = "he"
                label = 0
                distractor = random.choice(female_names)
            else:
                name = random.choice(female_names)
                pr = "she"
                label = 1
                distractor = random.choice(male_names)

            # 两句式：句末固定 because，模型预测下一 token（he/she）
            event = random.choice(TWO_SENTENCE_EVENTS)
            intro = random.choice(TWO_SENTENCE_INTROS)
            verb = random.choice(TWO_SENTENCE_VERBS)
            verb_anchor = verb
            prefix = f"{name} and {distractor} {event}. {intro} {name} {verb} because"
            if pr == "he":
                corr_name = random.choice(female_names)
            else:
                corr_name = random.choice(male_names)
            corrupted = f"{corr_name} and {distractor} {event}. {intro} {corr_name} {verb} because"

            # 确保 clean/corrupted 在 tokenizer 下 token 长度完全一致（避免 EAP-IG 报错）
            clean_len = len(self.tokenizer(prefix, add_special_tokens=False).input_ids)
            corr_len = len(self.tokenizer(corrupted, add_special_tokens=False).input_ids)
            if clean_len != corr_len:
                continue

            res.append(
                AgrGenderSample(
                    clean=prefix,
                    corrupted=corrupted,
                    label=label,
                    S=name,
                    D=distractor,
                    PR=pr,
                    verb=verb,
                    verb_anchor=verb_anchor,
                    template="two_sentence_because",
                )
            )
        return res

    def _from_records(self, records: List[Dict[str, Any]]) -> List[AgrGenderSample]:
        res = []
        for r in records:
            # 支持两种字段命名：
            # - CSV/原始: clean/corrupted
            # - 标准导出: text/corrupted_text
            if "clean" in r:
                clean = str(r["clean"]).strip()
            else:
                clean = str(r.get("text", "")).strip()
            if "corrupted" in r:
                corrupted = str(r["corrupted"]).strip()
            else:
                corrupted = str(r.get("corrupted_text", "")).strip()
            if not clean or not corrupted:
                raise ValueError("记录缺少 clean/corrupted（或 text/corrupted_text）字段")
            label = int(r.get("label", 0))
            S = str(r.get("S", clean.split()[0] if clean else "")).strip()
            # D：尽量从字段读取，否则从 "A and B ..." 解析
            D = str(r.get("D", "")).strip()
            if not D:
                parts = clean.split()
                # 形如 "A and B ..."
                if len(parts) >= 3 and parts[1].lower() == "and":
                    D = parts[2].strip().strip(",")
            PR = str(r.get("PR", ("he" if label == 0 else "she"))).strip().lower()
            # 优先使用记录里已有的 verb 信息；否则从文本解析
            verb = str(r.get("verb", "")).strip()
            verb_anchor = str(r.get("verb_anchor", "")).strip()
            if not verb:
                # 从文本解析动词（粗略）：取主语后到 because/after/... 之前
                words = clean.split()
                if len(words) > 2:
                    # 尝试找到连词位置
                    stop_words = {"because", "after", "while", "when", "although"}
                    try:
                        stop_idx = next(i for i,w in enumerate(words) if w in stop_words)
                    except StopIteration:
                        stop_idx = len(words)
                    verb = " ".join(words[1:stop_idx]) if stop_idx > 1 else (words[1] if len(words) > 1 else "")
                else:
                    verb = words[1] if len(words) > 1 else ""
            if not verb_anchor and verb:
                verb_anchor = verb.split()[0]
            res.append(
                AgrGenderSample(
                    clean=clean,
                    corrupted=corrupted,
                    label=label,
                    S=S,
                    D=D,
                    PR=PR,
                    verb=verb,
                    verb_anchor=verb_anchor,
                    template=r.get("template", "from_csv"),
                )
            )
        return res

    # ---- Dataset 接口 ----
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.toks[idx], self.corrupted_toks[idx], self.samples[idx].label

    def to(self, device: str):
        self._device = device
        return super().to(device)

    # ---- 兼容 IOIDataset 的辅助 ----
    def gen_flipped_prompts(self, flip=None):
        swapped = []
        for s in self.samples:
            new_pr = "she" if s.PR.lower() == "he" else "he"
            swapped.append(
                AgrGenderSample(
                    clean=s.corrupted,
                    corrupted=s.clean,
                    label=1 - s.label,
                    S=s.corrupted.split()[0],
                    D=s.D,
                    PR=new_pr,
                    verb=s.verb,
                    verb_anchor=s.verb_anchor,
                    template=s.template + "_flipped",
                )
            )
        return AgrGenderDataset(
            tokenizer=self.tokenizer,
            samples=swapped,
            prepend_bos=self.prepend_bos,
            device=self.device,
        )

    def logit_diff(self, logits: t.Tensor, mean: bool = True, loss: bool = False):
        """
        logits: [batch, seq, vocab]
        使用末位 logits，计算 he/she 的差，按 label 决定方向。
        """
        pos_logits = logits[t.arange(logits.size(0)), self.input_lengths - 1]
        probs = t.softmax(pos_logits, dim=-1)
        he_probs = probs[:, self.he_token_id]
        she_probs = probs[:, self.she_token_id]
        labels = t.tensor([s.label for s in self.samples], device=logits.device)
        results = t.where(labels == 0, he_probs - she_probs, she_probs - he_probs)
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results
