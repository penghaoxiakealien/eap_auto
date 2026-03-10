from ioi_dataset import NAMES, IOIDataset
import os
import re
import json
from rich import print as rprint
from rich.table import Table
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
device = t.device('cuda')

N = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device),
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")


def format_prompt(sentence: str) -> str:
    """Format a prompt by underlining names (for rich print)"""
    return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    """Makes and displays a table, from cols rather than rows (using rich print)"""
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)

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


# make_table(
#     colnames=["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
#     cols=[
#         map(format_prompt, ioi_dataset.sentences),
#         model.to_string(ioi_dataset.s_tokenIDs).split(),
#         model.to_string(ioi_dataset.io_tokenIDs).split(),
#         map(format_prompt, abc_dataset.sentences),
#     ],
#     title="Sentences from IOI vs ABC distribution",
# )

model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff(abc_logits_original).item()

# print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
# print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

# make_table(
#     colnames=["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
#     cols=[
#         map(format_prompt, ioi_dataset.sentences),
#         ioi_per_prompt_diff,
#         map(format_prompt, abc_dataset.sentences),
#         abc_per_prompt_diff,
#     ],
#     title="Sentences from IOI vs ABC distribution",
# )
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


print(f"IOI metric (IOI dataset): {ioi_metric(ioi_logits_original):.4f}")
print(f"IOI metric (ABC dataset): {ioi_metric(abc_logits_original):.4f}")

def patch_or_freeze_head_vectors(
    orig_head_vector,
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
):
    """
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them to their values in
    orig_cache), except for head_to_patch (if it's in this layer) which we patch with the value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: ActivationCache | None = abc_cache,
    orig_cache: ActivationCache | None = ioi_cache,
) :
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks, names_filter=z_name_filter, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Looping over every possible sender head (the receiver is always the final resid_post)
    for sender_layer, sender_head in tqdm(list(product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen

        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks, names_filter=resid_post_name_filter, return_type=None
        )
        # if (sender_layer, sender_head) == (9, 9):
        #     return patched_cache
        assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results


path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric)
fig=imshow(
    100 * path_patch_head_to_final_resid_post,
    title="Direct effect on logit difference",
    labels={"x": "Head", "y": "Layer", "color": "Logit diff. variation"},
    coloraxis=dict(colorbar_ticksuffix="%"),
    width=600,
    return_fig=True,
)
fig.write_image("results/ioi/path_patching/heads_direct_effect_on_logit_difference.png")


# Ensure the directory exists before saving the JSON file
os.makedirs(os.path.dirname("results/ioi/path_patching/heads_direct_effect_on_logit_difference.json"), exist_ok=True)

result = {}
for i, row in enumerate(path_patch_head_to_final_resid_post):
    for j, value in enumerate(row):
        key = f"{i}.{j}"
        result[key] = round(float(value) * 100, 2)

# 保存为 JSON 格式字符串
json_result = json.dumps(result, indent=4)

with open("results/ioi/path_patching/heads_direct_effect_on_logit_difference.json", "w") as f:
    f.write(json_result)