#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from eap.attribute import attribute
from eap.evaluate import evaluate_baseline, evaluate_graph
from eap.graph import Graph
from graph_collapse import (
    collapse_graph,
    load_graph,
    render_collapsed_graph,
    to_matrix,
    write_outputs,
)


def collate_eap(samples: List[Tuple[str, str, List[int]]]):
    clean, corrupted, labels = zip(*samples)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels


def _token_to_id(tokenizer, raw: str) -> int:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"非法 token 值: {raw!r}")
    candidates = [raw]
    if not raw.startswith(" "):
        candidates.append(" " + raw)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"token 无法单 token 化: {raw!r} (尝试 {candidates})")


class GardenDataset(Dataset):
    def __init__(self, filepath: Path, tokenizer):
        self.df = pd.read_csv(filepath)
        self.tokenizer = tokenizer

        for col in ["clean", "corrupted"]:
            if col not in self.df.columns:
                raise ValueError(f"缺少列: {col}")

        if "gpt2_correct_idx" in self.df.columns and "gpt2_incorrect_idx" in self.df.columns:
            self.df["correct_id"] = self.df["gpt2_correct_idx"].astype(int)
            self.df["incorrect_id"] = self.df["gpt2_incorrect_idx"].astype(int)
        else:
            if "correct_token" not in self.df.columns or "incorrect_token" not in self.df.columns:
                raise ValueError("缺少 correct_token / incorrect_token 列")
            self.df["correct_id"] = self.df["correct_token"].apply(lambda x: _token_to_id(tokenizer, x))
            self.df["incorrect_id"] = self.df["incorrect_token"].apply(lambda x: _token_to_id(tokenizer, x))

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return row["clean"], row["corrupted"], [row["correct_id"], row["incorrect_id"]]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_eap)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
    loss: bool = False,
):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run EAP-IG on garden NPZ v-trans (mod) with GPT2.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条样本调试。")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--topn", type=int, default=800)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--drop-input", action="store_true")
    parser.add_argument("--drop-logits", action="store_true")
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if "cuda" in device else torch.float32

    def _load_model(primary: str, local_path: Optional[str]):
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                local_path,
                local_files_only=True,
                torch_dtype=dtype,
            )
            official_name = primary if primary in {"gpt2", "gpt2-small"} else primary
            if official_name == "gpt2-small":
                official_name = "gpt2"
            mdl = HookedTransformer.from_pretrained(
                official_name,
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device=device,
                dtype=dtype,
                tokenizer=tokenizer,
                hf_model=hf_model,
                local_files_only=True,
            )
            return mdl, local_path

        try:
            mdl = HookedTransformer.from_pretrained(
                primary,
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device=device,
                dtype=dtype,
            )
            return mdl, primary
        except (OSError, FileNotFoundError):
            if primary == "gpt2-small":
                mdl = HookedTransformer.from_pretrained(
                    "gpt2",
                    center_writing_weights=False,
                    center_unembed=False,
                    fold_ln=False,
                    device=device,
                    dtype=dtype,
                )
                return mdl, "gpt2"
            raise

    model, model_name_used = _load_model(args.model_name, args.model_path)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    dataset = GardenDataset(args.dataset, tokenizer=model.tokenizer)
    if args.shuffle:
        dataset.shuffle()
    if args.limit:
        dataset.head(args.limit)
    dataloader = dataset.to_dataloader(args.batch_size)

    metric_loss = partial(logit_diff, loss=True, mean=True)
    metric_eval = partial(logit_diff, loss=False, mean=False)

    graph = Graph.from_model(model)
    attribute(
        model,
        graph,
        dataloader,
        metric_loss,
        method="EAP-IG-inputs",
        ig_steps=args.ig_steps,
    )
    graph.apply_topn(args.topn, absolute=True)

    graph_json = output_dir / "graph.json"
    graph_pt = output_dir / "graph.pt"
    graph_png = output_dir / "graph.png"
    graph.to_json(str(graph_json))
    graph.to_pt(str(graph_pt))
    graph.to_graphviz(str(graph_png))

    baseline = evaluate_baseline(model, dataloader, metric_eval).mean().item()
    circuit_perf = evaluate_graph(model, graph, dataloader, metric_eval).mean().item()
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(
            {
                "model": model_name_used,
                "baseline_mean_diff": baseline,
                "circuit_mean_diff": circuit_perf,
                "topn": args.topn,
                "ig_steps": args.ig_steps,
                "batch_size": args.batch_size,
                "limit": args.limit,
            },
            fh,
            indent=2,
        )

    collapse_data = load_graph(graph_json)
    include_input = not args.drop_input
    include_logits = not args.drop_logits
    nodes, adjacency = collapse_graph(collapse_data, include_input, include_logits)
    matrix, edges = to_matrix(nodes, adjacency)
    collapse_prefix = output_dir / "graph_collapsed"
    json_path, csv_path, edges_path = write_outputs(
        collapse_prefix,
        nodes,
        matrix,
        edges,
        include_input,
        include_logits,
    )
    collapse_png = collapse_prefix.with_suffix(".png")
    render_collapsed_graph(nodes, edges, collapse_png)

    print(f"Baseline mean diff: {baseline:.4f}")
    print(f"Circuit mean diff: {circuit_perf:.4f}")
    print(f"Graph JSON: {graph_json}")
    print(f"Graph PT: {graph_pt}")
    print(f"Graph PNG: {graph_png}")
    print(f"Collapsed JSON: {json_path}")
    print(f"Collapsed CSV: {csv_path}")
    print(f"Collapsed edges CSV: {edges_path}")
    print(f"Collapsed PNG: {collapse_png}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
