import torch as t
from functools import partial
from transformer_lens import HookedTransformer, utils
from ioi_dataset import IOIDataset

# -----------------------------------------------------------------------------
# 1. 模型 & 数据
# -----------------------------------------------------------------------------
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True

N = 150
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device),
)

# -----------------------------------------------------------------------------
# 2. Metric 定义
# -----------------------------------------------------------------------------
def logits_to_logit_diff(logits, dataset: IOIDataset):
    io_logits = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.io_tokenIDs]
    s_logits  = logits[range(logits.size(0)), dataset.word_idx["end"], dataset.s_tokenIDs]
    return (io_logits - s_logits).mean().item()

# baseline
clean_logits = model(ioi_dataset.toks)
clean_logit_diff = logits_to_logit_diff(clean_logits, ioi_dataset)
print(f"✅ Clean baseline logit diff = {clean_logit_diff:.4f}")

# -----------------------------------------------------------------------------
# 3. Knock-out 函数
# -----------------------------------------------------------------------------
def knockout_heads(
    head_output: t.Tensor,  # [batch, seq, n_heads, d_head]
    hook,
    heads_to_knockout: list[tuple[int, int]]
):
    layer = hook.layer()
    for (l, h) in heads_to_knockout:
        if l == layer:
            head_output[:, :, h, :] = 0.0
    return head_output

# -----------------------------------------------------------------------------
# 4. 运行 knock-out 实验
# -----------------------------------------------------------------------------
def run_knockout(heads_to_knockout):
    model.reset_hooks()
    hook_fn = partial(knockout_heads, heads_to_knockout=heads_to_knockout)
    z_filter = lambda name: name.endswith("z")  # 注意：作用在 attention head 输出上
    logits = model.run_with_hooks(ioi_dataset.toks, fwd_hooks=[(z_filter, hook_fn)])
    return logits_to_logit_diff(logits, ioi_dataset)

# -----------------------------------------------------------------------------
# 5. 测试
# -----------------------------------------------------------------------------
nmh = [(9, 6), (9, 9), (10, 0)]

# 单个头
for h in nmh:
    diff = run_knockout([h])
    print(f"❌ Knockout head {h[0]}.{h[1]} → logit diff = {diff:.4f} (Δ {diff - clean_logit_diff:.4f})")

# 全部头
diff_all = run_knockout(nmh)
print(f"\n❌ Knockout all NMH {nmh} → logit diff = {diff_all:.4f} (Δ {diff_all - clean_logit_diff:.4f})")
