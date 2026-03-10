#!/usr/bin/env python3
"""
通用任务数据接口，便于在 edge path patching 等脚本中解耦具体任务。
实现者需提供：
  - toks / corrupted_toks: [batch, seq] LongTensor
  - input_lengths: 每条样本的有效长度（预测位置为 input_length-1，除非另有 pred_positions）
  - pos_token_ids / neg_token_ids: 正/负 token id（可为列表/张量，长度为 batch 或全局统一）
  - logit_diff(logits, mean=True, loss=False): 根据任务定义返回分数
可选：
  - gen_flipped_prompts(): 返回一个 clean/corrupted 对调的数据集
  - word_idx: 若需要 attention 位置，可暴露
"""
from __future__ import annotations
from typing import Optional
import torch as t


class TaskDatasetBase:
    """任务数据基类，具体任务需要实现下列属性/方法。"""

    # 必需属性
    toks: t.Tensor                # [batch, seq]
    corrupted_toks: t.Tensor      # [batch, seq]
    input_lengths: t.Tensor       # [batch]
    pos_token_ids: t.Tensor       # [batch] or scalar
    neg_token_ids: t.Tensor       # [batch] or scalar

    def logit_diff(self, logits: t.Tensor, mean: bool = True, loss: bool = False) -> t.Tensor:
        """返回 logit 差分（正-负）。子类需实现。"""
        raise NotImplementedError

    # 可选
    def gen_flipped_prompts(self) -> "TaskDatasetBase":
        """返回 clean/corrupted 对调的数据集（若适用）。"""
        raise NotImplementedError

    @property
    def device(self):
        return self.toks.device

    def to(self, device: str):
        """迁移到指定设备。"""
        self.toks = self.toks.to(device)
        self.corrupted_toks = self.corrupted_toks.to(device)
        self.input_lengths = self.input_lengths.to(device)
        self.pos_token_ids = self.pos_token_ids.to(device)
        self.neg_token_ids = self.neg_token_ids.to(device)
        return self
