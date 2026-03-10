import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Optional
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

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
import sys
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "tests/experiments"))

from agr_gender_dataset import AgrGenderDataset


def collate_agr(samples):
    clean, corrupted, labels = zip(*samples)
    return list(clean), list(corrupted), torch.tensor(labels)


def parse_args():
    p = argparse.ArgumentParser(description="Run EAP-IG on agr_gender with GPT2-small.")
    p.add_argument("--data-path", type=Path, default=Path("datasets/agr_gender_eap_data.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/agr_gender_gpt2"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条样本调试。")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--topn", type=int, default=300)
    p.add_argument("--ig-steps", type=int, default=5)
    p.add_argument("--device", type=str, default=None, help="cuda/cuda:0/cpu；默认自动检测。")
    p.add_argument("--model-name", type=str, default="gpt2", help="HookedTransformer 模型名称。")
    p.add_argument("--model-path", type=str, default=None, help="本地模型目录（优先）。")
    return p.parse_args()


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def gender_logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    he_id: int,
    she_id: int,
    mean: bool = True,
    loss: bool = False,
):
    pos_logits = get_logit_positions(logits, input_length)
    probs = torch.softmax(pos_logits, dim=-1)
    he_probs = probs[:, he_id]
    she_probs = probs[:, she_id]
    labels = labels.to(logits.device)
    results = torch.where(labels == 0, he_probs - she_probs, she_probs - he_probs)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据集
    tokenizer = AutoTokenizer.from_pretrained(args.model_path or args.model_name, use_fast=True)
    ds = AgrGenderDataset(tokenizer=tokenizer, data_path=args.data_path)
    samples = [(s.clean, s.corrupted, s.label) for s in ds.samples]
    if args.shuffle:
        random.shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]
    dataloader = DataLoader(samples, batch_size=args.batch_size, collate_fn=collate_agr)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if "cuda" in device else torch.float32

    # 模型加载（本地优先）
    def _load_model(primary: str, local_path: Optional[str]):
        if local_path:
            tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                local_path, local_files_only=True, torch_dtype=dtype
            )
            mdl = HookedTransformer.from_pretrained(
                primary if primary != "gpt2-small" else "gpt2",
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
                device=device,
                dtype=dtype,
                tokenizer=tok,
                hf_model=hf_model,
                local_files_only=True,
            )
            return mdl, local_path
        mdl = HookedTransformer.from_pretrained(
            primary if primary != "gpt2-small" else "gpt2",
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device=device,
            dtype=dtype,
        )
        return mdl, primary

    model, model_name_used = _load_model(args.model_name, args.model_path)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    he_id, she_id = ds.he_token_id, ds.she_token_id
    metric_loss = partial(gender_logit_diff, he_id=he_id, she_id=she_id, loss=True, mean=True)
    metric_eval = partial(gender_logit_diff, he_id=he_id, she_id=she_id, loss=False, mean=False)

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
    nodes, adjacency = collapse_graph(collapse_data, include_input=True, include_logits=True)
    matrix, edges = to_matrix(nodes, adjacency)
    collapse_prefix = output_dir / "graph_collapsed"
    json_path, csv_path, edges_path = write_outputs(
        collapse_prefix,
        nodes,
        matrix,
        edges,
        include_input=True,
        include_logits=True,
    )
    collapse_png = collapse_prefix.with_suffix(".png")
    render_collapsed_graph(nodes, edges, collapse_png)

    print(f"Baseline mean diff: {baseline:.4f}")
    print(f"Circuit mean diff: {circuit_perf:.4f}")
    print(f"Graph JSON: {graph_json}")
    print(f"Graph PT: {graph_pt}")
    print(f"Graph PNG: {graph_png}")
    print(f"Collapsed JSON: {json_path}")
    print(f"Collapsed PNG: {collapse_png}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
