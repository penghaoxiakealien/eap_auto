from ioi_dataset import IOIDataset
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformer_lens import HookedTransformer
from typing import Callable
from itertools import product
from functools import partial
from tqdm import tqdm
import torch as t
import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from plotly_utils import imshow
from ioi_dataset import IOIDataset
from ioi_dataset import NAMES, IOIDataset
from heads_contribution_to_logits import patch_or_freeze_head_vectors

torch.cuda.empty_cache()
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
device = torch.device('cuda')

N = 2
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device),
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

def logits_to_ave_logit_diff(
    logits, ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False
):
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits= logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
    ]
    s_logits = logits[
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
    ]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits

    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff(abc_logits_original).item()

def ioi_metric(
    logits,
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    """
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    """
    patched_logit_diff = logits_to_ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


def patch_head_input(
    orig_activation,
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: list[tuple[int, int]],
):
    """
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    """
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation


def get_path_patch_head_to_heads(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: ActivationCache | None = None,
    orig_cache: ActivationCache | None = None,
) :
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input, for example if receiver_input = "v" and
    receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)], we're doing path patching from each head to the value inputs of
    the S-inhibition heads.

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks, names_filter=z_name_filter, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks, names_filter=receiver_hook_names_filter, return_type=None
        )
        # model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        # ========== Step 3 ==========
        # Run on x_orig, patching in the receiver node(s) from the previously cached value

        hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,
            head_list=receiver_heads,
        )
        patched_logits = model.run_with_hooks(
            orig_dataset.toks, fwd_hooks=[(receiver_hook_names_filter, hook_fn)], return_type="logits"
        )

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results


model.reset_hooks()

s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
    receiver_heads=[(9, 6), (9, 9), (10, 0)], receiver_input="q", model=model, patching_metric=ioi_metric
)
print("name_mover_head_path_patching_results:", s_inhibition_value_path_patching_results)
fig=imshow(
    100 * s_inhibition_value_path_patching_results,
    title="Direct effect on Name Mover Heads' queries",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
    width=600,
    coloraxis=dict(colorbar_ticksuffix="%"),
    return_fig=True,
)
fig.write_image(f"results/ioi/path_patching/name_mover_head_query_path_patching_results.png")