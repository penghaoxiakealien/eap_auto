#!/usr/bin/env python
"""
edge_path_patching.py

Compute path-patching contributions for every edge in a collapsed EAP graph.
Handles both attention→attention edges (q/k/v channels) and attention→logits edges.

Example:
    python edge_path_patching.py \
        --collapsed-graph results/ioi_gpt2/graph_collapsed.json \
        --edge-weights results/ioi_gpt2/patch_graph.json \
        --prompt-type mixed \
        --dataset-size 64 \
        --default-input q \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import torch as t
from tqdm import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from ioi_dataset import IOIDataset


# --------------------------------------------------------------------------- #
# Dataclasses and parsing helpers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HeadRef:
    layer: int
    head: int

    @classmethod
    def from_name(cls, name: str) -> "HeadRef":
        layer_str, head_str = name.split(".")
        return cls(layer=int(layer_str[1:]), head=int(head_str[1:]))

    def as_name(self) -> str:
        return f"a{self.layer}.h{self.head}"


@dataclass(frozen=True)
class EdgeSpec:
    src: HeadRef
    target_kind: Literal["attention", "logits"]
    dst: Optional[HeadRef] = None
    receiver_input: Optional[str] = None

    def label(self) -> str:
        if self.target_kind == "attention":
            return f"{self.src.as_name()}->{self.dst.as_name()}<{self.receiver_input}>"
        return f"{self.src.as_name()}->logits"


def load_collapsed_edges(collapsed_path: Path) -> List[Tuple[str, str]]:
    data = json.loads(collapsed_path.read_text())
    edges = data.get("edge_list")
    if edges is None:
        raise ValueError(f"{collapsed_path} missing 'edge_list'")
    return [(src, dst) for src, dst in edges]


def load_edge_specs(
    collapsed_graph: Path,
    default_input: str,
    full_graph: Optional[Path] = None,
) -> List[EdgeSpec]:
    valid_inputs = {"q", "k", "v"}
    if default_input not in valid_inputs:
        raise ValueError(f"--default-input must be one of {valid_inputs}")

    edges = load_collapsed_edges(collapsed_graph)
    qkv_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    if full_graph:
        raw = json.loads(full_graph.read_text()).get("edges", {})
        for info in raw.values():
            if not info.get("in_graph", False):
                continue
            parent = info["parent"]
            child = info["child"]
            receiver = info.get("qkv")
            if receiver:
                qkv_lookup[(parent, child)].append(receiver)

    specs: List[EdgeSpec] = []
    for src, dst in edges:
        if not src.startswith("a"):
            continue

        if dst == "logits":
            specs.append(
                EdgeSpec(
                    src=HeadRef.from_name(src),
                    target_kind="logits",
                )
            )
        elif dst.startswith("a"):
            receivers = qkv_lookup.get((src, dst)) or [default_input]
            for receiver in receivers:
                specs.append(
                    EdgeSpec(
                        src=HeadRef.from_name(src),
                        dst=HeadRef.from_name(dst),
                        target_kind="attention",
                        receiver_input=receiver,
                    )
                )

    return specs


# --------------------------------------------------------------------------- #
# Path patching utilities
# --------------------------------------------------------------------------- #

def logits_to_logit_diff(logits: t.Tensor, dataset: IOIDataset, per_prompt: bool = False) -> t.Tensor:
    io_logits = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.s_tokenIDs]
    diff = io_logits - s_logits
    return diff if per_prompt else diff.mean()


def patch_or_freeze_head_vectors(
    head_output: t.Tensor,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int],
) -> t.Tensor:
    head_output[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        head_output[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return head_output


def patch_head_input(
    activation: t.Tensor,
    hook: HookPoint,
    patched_cache: ActivationCache,
    receiver_heads: Sequence[HeadRef],
) -> t.Tensor:
    head_indices = [ref.head for ref in receiver_heads if ref.layer == hook.layer()]
    if head_indices:
        activation[:, :, head_indices] = patched_cache[hook.name][:, :, head_indices]
    return activation


def path_patch_to_head(
    model: HookedTransformer,
    receiver_head: HeadRef,
    receiver_input: str,
    orig_dataset: IOIDataset,
    new_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    receiver_layer = receiver_head.layer
    z_filter = lambda name: name.endswith("z")
    receiver_hook = utils.get_act_name(receiver_input, receiver_layer)
    receiver_filter = lambda name: name == receiver_hook

    results = t.zeros(receiver_layer, model.cfg.n_heads, device="cpu", dtype=t.float32)

    for sender_layer in range(receiver_layer):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            hook_fn = lambda tensor, hook: patch_or_freeze_head_vectors(
                tensor,
                hook,
                new_cache=new_cache,
                orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            )
            model.add_hook(z_filter, hook_fn, level=1)

            with t.no_grad():
                _, recv_cache = model.run_with_cache(
                    orig_dataset.toks, names_filter=receiver_filter, return_type=None
                )

            model.reset_hooks()
            head_hook = lambda tensor, hook: patch_head_input(
                tensor, hook, patched_cache=recv_cache, receiver_heads=[receiver_head]
            )
            with t.no_grad():
                patched_logits = model.run_with_hooks(
                    orig_dataset.toks,
                    fwd_hooks=[(receiver_filter, head_hook)],
                    return_type="logits",
                )

            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta

    model.reset_hooks()
    return results


def path_patch_to_residual(
    model: HookedTransformer,
    orig_dataset: IOIDataset,
    new_dataset: IOIDataset,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
    baseline: float,
) -> t.Tensor:
    z_filter = lambda name: name.endswith("z")

    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device="cpu", dtype=t.float32)

    resid_post_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_filter = lambda name: name == resid_post_name

    for sender_layer in range(model.cfg.n_layers):
        for sender_head in range(model.cfg.n_heads):
            model.reset_hooks()
            hook_fn = lambda tensor, hook: patch_or_freeze_head_vectors(
                tensor,
                hook,
                new_cache=new_cache,
                orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            )
            model.add_hook(z_filter, hook_fn)

            with t.no_grad():
                _, cache = model.run_with_cache(
                    orig_dataset.toks, names_filter=resid_filter, return_type=None
                )

            patched_logits = model.unembed(model.ln_final(cache[resid_post_name]))
            delta = logits_to_logit_diff(patched_logits, orig_dataset).item() - baseline
            results[sender_layer, sender_head] = delta

    model.reset_hooks()
    return results


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_edge_path_patching(
    collapsed_graph: Path,
    output_path: Path,
    prompt_type: str,
    dataset_size: int,
    seed: int,
    default_input: str,
    device: str,
    full_graph: Optional[Path] = None,
) -> None:
    specs = load_edge_specs(collapsed_graph, default_input, full_graph)
    if not specs:
        raise ValueError("No attention or logits edges detected in collapsed graph.")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ioi_dataset = IOIDataset(
        prompt_type=prompt_type,
        N=dataset_size,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=device,
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    with t.no_grad():
        clean_logits = model(ioi_dataset.toks)
        baseline = logits_to_logit_diff(clean_logits, ioi_dataset).item()

    z_filter = lambda name: name.endswith("z")
    with t.no_grad():
        _, orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_filter, return_type=None)
        _, new_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_filter, return_type=None)

    # Precompute residual patching results if needed
    logits_needed = any(spec.target_kind == "logits" for spec in specs)
    residual_matrix = None
    if logits_needed:
        residual_matrix = path_patch_to_residual(
            model=model,
            orig_dataset=ioi_dataset,
            new_dataset=abc_dataset,
            orig_cache=orig_cache,
            new_cache=new_cache,
            baseline=baseline,
        )

    # Group receiver heads for attention-focused patching
    attention_groups: Dict[Tuple[HeadRef, str], List[EdgeSpec]] = defaultdict(list)
    for spec in specs:
        if spec.target_kind == "attention":
            attention_groups[(spec.dst, spec.receiver_input)].append(spec)

    results: List[Dict[str, object]] = []

    # Process attention receivers
    for (receiver_head, receiver_input), group in tqdm(attention_groups.items(), desc="Attention receivers"):
        sender_matrix = path_patch_to_head(
            model=model,
            receiver_head=receiver_head,
            receiver_input=receiver_input,
            orig_dataset=ioi_dataset,
            new_dataset=abc_dataset,
            orig_cache=orig_cache,
            new_cache=new_cache,
            baseline=baseline,
        )
        for spec in group:
            delta = float(sender_matrix[spec.src.layer, spec.src.head])
            results.append(
                {
                    "edge": spec.label(),
                    "receiver_kind": "attention",
                    "src": spec.src.as_name(),
                    "dst": spec.dst.as_name(),
                    "receiver_input": spec.receiver_input,
                    "delta_logit_diff": delta,
                    "patched_logit_diff": baseline + delta,
                }
            )

    # Process logits receivers
    if logits_needed:
        for spec in specs:
            if spec.target_kind != "logits":
                continue
            delta = float(residual_matrix[spec.src.layer, spec.src.head])
            results.append(
                {
                    "edge": spec.label(),
                    "receiver_kind": "logits",
                    "src": spec.src.as_name(),
                    "dst": "logits",
                    "receiver_input": None,
                    "delta_logit_diff": delta,
                    "patched_logit_diff": baseline + delta,
                }
            )

    payload = {
        "meta": {
            "collapsed_graph": str(collapsed_graph),
            "full_graph": str(full_graph) if full_graph else None,
            "prompt_type": prompt_type,
            "dataset_size": dataset_size,
            "seed": seed,
            "baseline_logit_diff": baseline,
        },
        "edges": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(results)} edge scores to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path patch every edge (attention/logits) in a collapsed graph.")
    parser.add_argument("--collapsed-graph", type=Path, required=True, help="Collapsed JSON from graph_collapse.py")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON for edge weights")
    parser.add_argument("--full-graph", type=Path, help="Optional full graph to recover q/k/v labels")
    parser.add_argument("--prompt-type", type=str, default="mixed")
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--default-input", type=str, default="q", choices=["q", "k", "v"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_edge_path_patching(
        collapsed_graph=args.collapsed_graph,
        output_path=args.output,
        prompt_type=args.prompt_type,
        dataset_size=args.dataset_size,
        seed=args.seed,
        default_input=args.default_input,
        device=args.device,
        full_graph=args.full_graph,
    )


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()