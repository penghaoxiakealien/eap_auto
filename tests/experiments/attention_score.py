from transformer_lens import HookedTransformer
import os
import torch as t
from plotly_utils import imshow

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def compute_token_attention_means(model, input_sequence, device='cuda'):
    """
    计算输入序列中倒数第二个token的注意力均值。
    """
    model.reset_hooks()
    input_toks = model.tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False).to(device)
    print(f"Input tokens shape: {input_toks['input_ids'].shape}")  # [batch_size, seq_len]

    # 运行模型并存储激活
    outputs, cache = model.run_with_cache(input_toks['input_ids'], names_filter=lambda name: name.endswith('hook_pattern'))
    print(f"Model outputs shape: {outputs.shape}")  # [batch_size, seq_len, vocab_size]

    # 获取真实tokens
    real_tokens = model.tokenizer.convert_ids_to_tokens(input_toks['input_ids'][0])
    real_tokens = [tok.replace('Ġ', '') for tok in real_tokens]  # 移除 'Ġ'
    seq_len = input_toks['input_ids'].shape[1]
    real_length = len(real_tokens)
    print(f"Original token count: {real_length}, Seq len: {seq_len}")
    print(f"Real tokens: {real_tokens}")

    target_pos = real_length - 1
    target_token = real_tokens[target_pos]
    print(f"Analyzing attention from token at position {target_pos} ('{target_token}') to all other positions")

    # 初始化注意力均值张量
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    attention_means = t.zeros(num_layers, num_heads, real_length, device=device)
    print(f"Attention means tensor shape: {attention_means.shape}")  # [n_layers, n_heads, seq_len]

    for layer in range(num_layers):
        for head in range(num_heads):
            attention_scores = cache[f'blocks.{layer}.attn.hook_pattern'][:, head]
            print(f"Layer {layer} Head {head} attention scores shape: {attention_scores.shape}")  # [batch_size, seq_len, seq_len]

            # 从目标位置到所有其他位置的注意力
            target_attention = attention_scores[:, target_pos, :]
            print(f"Target attention shape: {target_attention.shape}")  # [batch_size, seq_len]

            attention_means[layer, head, :] = target_attention.mean(dim=0)
            print(f"Updated attention means shape: {attention_means.shape}")

    return attention_means, real_tokens, target_pos


def visualize_heads_with_imshow(attention_means, model, real_tokens, target_pos):
    """
    使用imshow可视化每个头的注意力模式。
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads

    # 为每个token添加位置索引
    labeled_tokens = []
    for i, tok in enumerate(real_tokens):
        if i == target_pos:
            labeled_tokens.append(f"→{i+1}:{tok}←")  # 用箭头标记目标token
        else:
            labeled_tokens.append(f"{i+1}:{tok}")

    # 创建目录
    os.makedirs("results/ioi/attention_scores", exist_ok=True)

    for layer in range(num_layers):
        layer_attention = attention_means[layer]  # [n_heads, seq_len]

        fig = imshow(
            layer_attention,
            x=labeled_tokens,  # 带位置前缀的tokens
            y=[f"Head {head}" for head in range(num_heads)],
            labels={
                "x": "Token Position (position:token)",
                "y": "Attention Head",
                "color": "Attention Score"
            },
            title=f"Layer {layer} - Attention from position {target_pos+1} ('{real_tokens[target_pos]}')",
            width=800 + len(real_tokens)*10,  # 根据序列长度动态调整宽度
            height=400,
            color_continuous_scale="Blues",
            zmin=0.0,
            zmax=1.0,
            xaxis_tickangle=45,  # 旋转标签以便阅读
            return_fig=True
        )

        fig.update_layout(
            margin=dict(l=100, r=100, t=100, b=150),  # 调整边距
            xaxis_tickfont=dict(size=10)  # 较小字体以适应长标签
        )

        # 保存图像
        fig.write_image(f"results/ioi/attention_scores/layer_{layer}_attention_from_pos_{target_pos}.png")
        print(f"Saved attention scores for Layer {layer} to results/ioi/attention_scores/layer_{layer}_attention_from_pos_{target_pos}.png")


# 加载模型
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
print(f"\nModel loaded: {model_name}")

# 指定输入序列
input_sequence = "When Victoria and Jane got a snack at the store, Jane decided to give it to"
print(f"\nInput sequence: {input_sequence}")

# 计算注意力均值并获取真实tokens
print("\nComputing attention means...")
attention_means, real_tokens, target_pos = compute_token_attention_means(model, input_sequence)
print(f"Attention means shape: {attention_means.shape}")  # [n_layers, n_heads, seq_len]
print(f"real_tokens shape: {len(real_tokens)}")
print(f"Analyzing attention from token at position {target_pos} ('{real_tokens[target_pos]}')")

# 使用imshow可视化
print("\nGenerating visualizations...")
visualize_heads_with_imshow(attention_means, model, real_tokens, target_pos)
print("\nVisualization complete!")