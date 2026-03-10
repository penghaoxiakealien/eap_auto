#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 Garden dominant 模式聚类后的注意力头进行路径插补分类。"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch as t
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from garden_dataset import GardenDataset, GardenSample

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class HeadMetric:
    head: str
    patched_logit_diff: float
    delta_logit_diff: float
    metric: float


def parse_heads(heads: Iterable[str]) -> List[Tuple[int, int]]:
    parsed: List[Tuple[int, int]] = []
    for value in heads:
        try:
            layer_str, head_str = value.split(".")
            parsed.append((int(layer_str), int(head_str)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"非法 head 标识: {value} (期望 L.H)") from exc
    return parsed


def compute_metric(
    patched_diff: float,
    clean_logit_diff: float,
    corrupted_logit_diff: float,
) -> float:
    denominator = clean_logit_diff - corrupted_logit_diff
    if abs(denominator) < 1e-9:
        return 0.0
    return (patched_diff - clean_logit_diff) / denominator


def patch_or_freeze_head_vectors(
    act: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    act.copy_(orig_cache[hook.name])
    if head_to_patch[0] == hook.layer():
        act[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return act


def load_model(model_name: str, model_path: Optional[Path], device: t.device) -> HookedTransformer:
    official_name = model_name if model_name != "gpt2-small" else "gpt2"
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        hf_model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
        model = HookedTransformer.from_pretrained(
            official_name,
            device=device,
            tokenizer=tokenizer,
            hf_model=hf_model,
            local_files_only=True,
        )
    else:
        candidates = [model_name]
        if model_name == "gpt2-small":
            candidates.append("gpt2")
        last_err: Optional[Exception] = None
        for name in candidates:
            try:
                model = HookedTransformer.from_pretrained(name, device=device)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            assert last_err is not None
            raise last_err
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def run_path_patching(
    model: HookedTransformer,
    dataset: GardenDataset,
    head: Tuple[int, int],
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    corrupted_cache: ActivationCache,
    clean_cache: ActivationCache,
) -> HeadMetric:
    model.reset_hooks()
    resid_post_hook = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_hook
    z_filter = lambda name: name.endswith("z")

    hook_fn = lambda act, hook: patch_or_freeze_head_vectors(
        act, hook, corrupted_cache, clean_cache, head
    )
    model.add_hook(z_filter, hook_fn)

    _, patched_cache = model.run_with_cache(dataset.toks, names_filter=resid_filter, return_type=None)
    patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook]))
    patched_diff = dataset.logit_diff(patched_logits, mean=True).item()
    head_name = f"{head[0]}.{head[1]}"
    delta = patched_diff - clean_logit_diff
    metric = compute_metric(patched_diff, clean_logit_diff, corrupted_logit_diff)

    return HeadMetric(head=head_name, patched_logit_diff=patched_diff, delta_logit_diff=delta, metric=metric)


def classify_metrics(metrics: List[HeadMetric]) -> Dict[str, List[str]]:
    groups = {"positive": [], "negative": []}
    for item in metrics:
        if item.metric >= 0:
            groups["positive"].append(item.head)
        else:
            groups["negative"].append(item.head)
    return groups


def update_soft_payload(
    soft_data: Dict[str, Any],
    metrics: List[HeadMetric],
    classification: Dict[str, List[str]],
    pattern_groups: Dict[str, List[str]],
    output_path: Path,
) -> None:
    per_head = (
        soft_data.get("patterns", {}).get("per_head")
        or soft_data.get("head_patterns")
        or soft_data.get("per_head")
    )
    if not isinstance(per_head, dict):  # pragma: no cover - defensive
        raise ValueError("soft JSON 中缺少 per_head 信息")

    metric_map = {item.head: item for item in metrics}
    class_map = {head: label for label, heads in classification.items() for head in heads}

    for head, info in per_head.items():
        details = metric_map.get(head)
        label = class_map.get(head, "unknown")
        payload = {
            "classification": label,
            "metric": round(details.metric, 6) if details else None,
            "delta_logit_diff": round(details.delta_logit_diff, 6) if details else None,
            "patched_logit_diff": round(details.patched_logit_diff, 6) if details else None,
        }
        info["path_patch"] = payload
        signature = info.get("signature")
        if signature and details:
            info["signature_path_patch"] = f"{signature} | PP:{label}"
        elif signature:
            info["signature_path_patch"] = f"{signature} | PP:NA"

    group_breakdown: Dict[str, Any] = {}
    for signature, heads in pattern_groups.items():
        counts = {label: 0 for label in classification}
        for head in heads:
            label = class_map.get(head, "unknown")
            if label in counts:
                counts[label] += 1
        group_breakdown[signature] = {
            "heads": heads,
            "counts": counts,
        }

    soft_data.setdefault("meta", {})["path_patch"] = {
        "classification": classification,
        "group_breakdown": group_breakdown,
    }
    soft_data["path_patch_results"] = {
        head: {
            "metric": metric_map[head].metric,
            "delta_logit_diff": metric_map[head].delta_logit_diff,
            "patched_logit_diff": metric_map[head].patched_logit_diff,
            "classification": class_map.get(head, "unknown"),
        }
        for head in sorted(metric_map.keys())
    }
    soft_data["path_patch_groups"] = group_breakdown

    output_path.write_text(json.dumps(soft_data, indent=2, ensure_ascii=False))


def _sample_dataset_samples(samples: List[GardenSample], n_prompts: int, seed: int) -> List[GardenSample]:
    if n_prompts <= 0 or n_prompts >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, n_prompts)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 Garden dominant 聚类后的注意力头进行路径插补分类")
    parser.add_argument("--dominant-soft", type=Path, required=True, help="thr75 soft JSON 路径")
    parser.add_argument("--output-json", type=Path, required=True, help="path patch 结果输出 JSON")
    parser.add_argument("--updated-soft", type=Path, required=True, help="写入 path patch 结果的新 soft JSON")
    parser.add_argument("--data-path", type=Path, required=True, help="Garden 数据集 CSV")
    parser.add_argument("--model-name", type=str, default="gpt2", help="HookedTransformer 模型名")
    parser.add_argument("--model-path", type=Path, help="本地模型路径，可选")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-prompts", type=int, default=50, help="Garden 样本数量")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def run_path_patch_analysis(
    dominant_soft: Path,
    output_json: Path,
    updated_soft: Path,
    data_path: Path,
    model_name: str = "gpt2",
    model_path: Optional[Path] = None,
    device: str = "cuda",
    n_prompts: int = 50,
    seed: int = 1,
) -> Dict[str, Any]:
    soft_data = json.loads(dominant_soft.read_text())
    patterns_block = soft_data.get("patterns", {})
    per_head = (
        patterns_block.get("per_head")
        or soft_data.get("head_patterns")
        or soft_data.get("per_head")
    )
    if not isinstance(per_head, dict):
        raise ValueError("dominant soft JSON 缺少 patterns.per_head")
    pattern_groups = (
        soft_data.get("pattern_groups")
        or patterns_block.get("groups_by_signature")
        or soft_data.get("groups_by_signature")
        or {}
    )
    if not isinstance(pattern_groups, dict):
        pattern_groups = {}

    head_names = sorted(per_head.keys())
    if not head_names:
        raise ValueError("未在 dominant soft JSON 中找到任何 head")

    head_tuples = parse_heads(head_names)
    dev = t.device(device)
    model = load_model(model_name, model_path, dev)
    t.manual_seed(seed)

    full_dataset = GardenDataset(
        tokenizer=model.tokenizer,
        data_path=data_path,
        prepend_bos=False,
        device=str(dev),
        seed=seed,
    )
    samples = _sample_dataset_samples(full_dataset.samples, n_prompts, seed)
    dataset = GardenDataset(
        tokenizer=model.tokenizer,
        samples=samples,
        prepend_bos=False,
        device=str(dev),
        seed=seed,
    )
    corrupted_dataset = dataset.gen_flipped_prompts()

    with t.no_grad():
        clean_logits, _ = model.run_with_cache(dataset.toks)
        corrupted_logits, _ = model.run_with_cache(corrupted_dataset.toks)

    clean_logit_diff = dataset.logit_diff(clean_logits, mean=True).item()
    corrupted_logit_diff = dataset.logit_diff(corrupted_logits, mean=True).item()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        _, corrupted_cache = model.run_with_cache(
            corrupted_dataset.toks, names_filter=z_filter, return_type=None
        )
        _, clean_cache = model.run_with_cache(
            dataset.toks, names_filter=z_filter, return_type=None
        )

    metrics: List[HeadMetric] = []
    for head in tqdm(head_tuples, desc="Path patching heads"):
        metric = run_path_patching(
            model,
            dataset,
            head,
            clean_logit_diff,
            corrupted_logit_diff,
            corrupted_cache,
            clean_cache,
        )
        metrics.append(metric)

    classification = classify_metrics(metrics)

    payload = {
        "meta": {
            "dominant_soft": str(dominant_soft),
            "data_path": str(data_path),
            "model_name": model_name,
            "model_path": str(model_path) if model_path else None,
            "device": device,
            "n_prompts": n_prompts,
            "seed": seed,
            "clean_logit_diff": clean_logit_diff,
            "corrupted_logit_diff": corrupted_logit_diff,
        },
        "metrics": {
            item.head: {
                "patched_logit_diff": item.patched_logit_diff,
                "delta_logit_diff": item.delta_logit_diff,
                "metric": item.metric,
            }
            for item in metrics
        },
        "classification": classification,
        "pattern_groups": pattern_groups,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"写入路径插补结果: {output_json}")

    updated = json.loads(dominant_soft.read_text())
    update_soft_payload(updated, metrics, classification, pattern_groups, updated_soft)
    print(f"写入带路径插补信息的 soft JSON: {updated_soft}")

    return {
        "payload": payload,
        "metrics": metrics,
        "classification": classification,
        "clean_logit_diff": clean_logit_diff,
        "corrupted_logit_diff": corrupted_logit_diff,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_path_patch_analysis(
        dominant_soft=args.dominant_soft,
        output_json=args.output_json,
        updated_soft=args.updated_soft,
        data_path=args.data_path,
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        n_prompts=args.n_prompts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
