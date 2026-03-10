#!/usr/bin/env python
"""
single_path_patch.py

Reusable utilities for *single‑head → single‑head* path patching on the Indirect Object Identification (IOI) task.

**Public API**
--------------
- `load_ioi_resources(...)` → `(model, ioi_dataset, abc_dataset)` – quickly load defaults
- `path_patch_single(model, ioi_dataset, abc_dataset, sender_head, receiver_head, ...)` → metric value
- `main(argv=None)` – programmatic entry‑point mirroring the CLI

Example
-------
```python
from single_path_patch import load_ioi_resources, path_patch_single, main

# Low‑level API
model, ioi_ds, abc_ds = load_ioi_resources()
metric = path_patch_single(model, ioi_ds, abc_ds, (9, 3), (10, 0))
print(metric)

# Call the CLI programmatically
main(["--sender_head", "9.3", "--receiver_head", "10.0", "--receiver_input", "q"])
```

Running as a script still works:
```bash
python single_path_patch.py --sender_head 9.3 --receiver_head 10.0 --receiver_input q
```
"""

from __future__ import annotations
import sys
sys.path.append("/data63/private/chensiyuan/EAP-IG/tests/experiments")
import argparse
import os
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import torch as t
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from ioi_dataset import IOIDataset
from heads_contribution_to_logits import patch_or_freeze_head_vectors
from plotly_utils import imshow

__all__ = [
    "load_ioi_resources",
    "path_patch_single",
    "main",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def str_to_head(head: str | Tuple[int, int]) -> Tuple[int, int]:
    """Convert "layer.head" or a tuple into `(layer, head)` ints."""
    if isinstance(head, tuple):
        return int(head[0]), int(head[1])
    try:
        layer_str, head_str = head.split(".")
        return int(layer_str), int(head_str)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"Head must be in format layer.head, e.g. '9.3' (got '{head}')."
        ) from exc


def logits_to_ave_logit_diff(logits, ioi_dataset: IOIDataset) -> t.Tensor:
    """Return the *average* logit difference `io - s`."""
    io_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
    ]
    return (io_logits - s_logits).mean()


def compute_ioi_metric(
    ioi_logits, abc_logits, ioi_dataset: IOIDataset
) -> Callable[[t.Tensor], float]:
    """Return a metric function mapping logits → [0, -1] range."""

    clean = logits_to_ave_logit_diff(ioi_logits, ioi_dataset).item()
    corrupted = logits_to_ave_logit_diff(abc_logits, ioi_dataset).item()

    def _metric(logits: t.Tensor) -> float:  # pylint: disable=invalid-name
        patched = logits_to_ave_logit_diff(logits, ioi_dataset)
        return (patched - clean) / (clean - corrupted)

    return _metric


# -----------------------------------------------------------------------------
# Resource loader (model + datasets)
# -----------------------------------------------------------------------------

def load_ioi_resources(
    model_name: str = "gpt2-small",
    device: str | t.device = "cuda",
    n_prompts: int = 2,
    seed: int = 1,
):
    """Return `(model, ioi_dataset, abc_dataset)` with sensible defaults."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    device = t.device(device)
    if device.type == "cuda":
        t.cuda.empty_cache()

    model = HookedTransformer.from_pretrained(model_name, device=device)

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=n_prompts,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(device),
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    return model, ioi_dataset, abc_dataset


# -----------------------------------------------------------------------------
# Core path‑patching routine
# -----------------------------------------------------------------------------

def path_patch_single(
    model: HookedTransformer,
    ioi_dataset: IOIDataset,
    abc_dataset: IOIDataset,
    sender_head: str | Tuple[int, int],
    receiver_head: str | Tuple[int, int],
    receiver_input: Literal["q", "k", "v"] = "q",
    metric_fn: Optional[Callable[[t.Tensor], float]] = None,
    save_fig: bool = False,
    fig_dir: str = "results/ioi/path_patching",
) -> float:
    """Path‑patch from `sender_head` → `receiver_head`.

    Returns the IOI metric (0 = clean, −1 = fully corrupted).
    Optionally saves a 1 × 1 heat‑map figure when `save_fig=True`.
    """

    sender_layer, sender_h = str_to_head(sender_head)
    receiver_layer, receiver_h = str_to_head(receiver_head)

    if sender_layer >= receiver_layer:
        raise ValueError("Sender layer must be < receiver layer for causal path.")

    # ---------------- Baseline metric ----------------
    if metric_fn is None:
        with t.no_grad():
            ioi_logits, _ = model.run_with_cache(ioi_dataset.toks)
            abc_logits, _ = model.run_with_cache(abc_dataset.toks)
        metric_fn = compute_ioi_metric(ioi_logits, abc_logits, ioi_dataset)

    # ---------------- Caches ----------------
    z_name = lambda n: n.endswith("z")
    model.reset_hooks()
    _, new_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_name, return_type=None)
    _, orig_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_name, return_type=None)

    # ---------------- Step 1: patch sender ----------------
    hook_sender = partial(
        patch_or_freeze_head_vectors,
        new_cache=new_cache,
        orig_cache=orig_cache,
        head_to_patch=(sender_layer, sender_h),
    )
    model.add_hook(z_name, hook_sender, level=1)

    # ---------------- Step 2: cache receiver input ----------------
    receiver_hook_name = utils.get_act_name(receiver_input, receiver_layer)
    receiver_filter = lambda n: n == receiver_hook_name
    _, patched_cache = model.run_with_cache(  # type: ignore
        ioi_dataset.toks, names_filter=receiver_filter, return_type=None
    )
    model.reset_hooks()

    # ---------------- Step 3: patch receiver input ----------------
    def patch_receiver(act: t.Tensor, hook: HookPoint):  # type: ignore
        act[:, :, receiver_h] = patched_cache[hook.name][:, :, receiver_h]
        return act

    patched_logits = model.run_with_hooks(
        ioi_dataset.toks,
        fwd_hooks=[(receiver_filter, patch_receiver)],
        return_type="logits",
    )

    metric = metric_fn(patched_logits).item()

    # ---------------- Optional figure ----------------
    if save_fig:
        os.makedirs(fig_dir, exist_ok=True)
        fig = imshow(
            t.tensor([[metric * 100]]),
            labels={"x": "Head", "y": "Layer", "color": "Metric (%)"},
            title="Single→single path patching",
            width=300,
            height=300,
            coloraxis=dict(colorbar_ticksuffix="%"),
            return_fig=True,
        )
        path = os.path.join(
            fig_dir,
            f"path_patch_{sender_layer}.{sender_h}_to_{receiver_layer}.{receiver_h}.png",
        )
        fig.write_image(path)
        print(f"[single_path_patch] Saved figure → {path}")

    return metric  # 0 = clean, −1 = corrupted


# -----------------------------------------------------------------------------
# CLI wrapper – callable from code or shell
# -----------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None):
    """Parse CLI args; pass a custom list to parse programmatically."""
    p = argparse.ArgumentParser(description="Single‑head path patching (IOI task)")
    p.add_argument("--sender_head", required=True, help="e.g. 9.3")
    p.add_argument("--receiver_head", required=True, help="e.g. 10.0")
    p.add_argument("--receiver_input", choices=["q", "k", "v"], default="q")
    p.add_argument("--model_name", default="gpt2-small")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_prompts", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_fig", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    """Entry‑point mirroring the CLI; accepts an argument list.

    Examples
    --------
    >>> main(["--sender_head", "9.3", "--receiver_head", "10.0"])
    >>> main()  # uses sys.argv[1:]
    """
    args = _parse_args(argv)
    model, ioi_ds, abc_ds = load_ioi_resources(
        model_name=args.model_name,
        device=args.device,
        n_prompts=args.n_prompts,
        seed=args.seed,
    )

    metric = path_patch_single(
        model,
        ioi_ds,
        abc_ds,
        args.sender_head,
        args.receiver_head,
        receiver_input=args.receiver_input,
        save_fig=args.save_fig,
    )
    print(f"Metric from {args.sender_head} → {args.receiver_head}: {metric:.3f}")
    return metric

if __name__ == "__main__":  # pragma: no cover
    main()
