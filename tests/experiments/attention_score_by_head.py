import argparse
from transformer_lens import HookedTransformer
import os
import torch as t
from plotly_utils import imshow
import json

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def compute_attention_scores(model, input_sequence, layer, head, device='cuda'):
    """
    计算特定层和特定头中，每个位置对其之前所有位置的注意力分数。
    """
    model.reset_hooks()
    input_toks = model.tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False).to(device)
    # print(f"input tokens: {input_toks['input_ids']}")
    # 运行模型并存储激活
    outputs, cache = model.run_with_cache(input_toks['input_ids'], names_filter=lambda name: name.endswith('hook_pattern'))

    # 获取真实tokens
    real_tokens = model.tokenizer.convert_ids_to_tokens(input_toks['input_ids'][0])
    real_tokens = [tok.replace('Ġ', '') for tok in real_tokens]  # 移除 'Ġ'
    

    # 获取特定层和头的注意力分数
    attention_scores = cache[f'blocks.{layer}.attn.hook_pattern'][:, head]  # [batch_size, seq_len, seq_len]
    # 移除 <|endoftext|> 标志
    if "<|endoftext|>" in real_tokens:
        start_idx = 1 if real_tokens[0] == "<|endoftext|>" else 0
        end_idx = -1 if real_tokens[-1] == "<|endoftext|>" else len(real_tokens)
        real_tokens = real_tokens[start_idx:end_idx]
        attention_scores = attention_scores[:, start_idx:end_idx, start_idx:end_idx]

    filtered_indices = [i for i, tok in enumerate(real_tokens) if tok != '']
    real_tokens = [real_tokens[i] for i in filtered_indices]
    attention_scores = attention_scores[:, filtered_indices, :][:, :, filtered_indices]
    # print(f"Filtered real tokens: {real_tokens}")
    # print(f"Raw attention scores shape for Layer {layer}, Head {head}: {attention_scores.shape}")

    print(f"Attention scores shape for Layer {layer}, Head {head}: {attention_scores.shape}")

    # 返回特定头的注意力分数和真实tokens
    return attention_scores[0], real_tokens  # 返回第一个batch的结果

def compute_raw_attention_scores(model, input_sequence, layer, head, device='cuda'):
    """
    计算特定层和特定头中，每个位置对其之前所有位置的未经过 Softmax 的注意力分数。
    """
    model.reset_hooks()
    input_toks = model.tokenizer(input_sequence, return_tensors="pt", add_special_tokens=False).to(device)
    print(f"input tokens: {input_toks['input_ids']}")
    # 定义一个钩子函数来捕获未经过 Softmax 的注意力 logits
    raw_attention_scores = None

    def hook_fn(value, hook):
        nonlocal raw_attention_scores
        raw_attention_scores = value[:, head]  # 提取特定头的注意力 logits

    # 注册钩子到指定层的注意力分数
    hook_name = f'blocks.{layer}.attn.hook_attn_scores'  # 修改为正确的钩子名称
    model.add_hook(hook_name, hook_fn)

    # 运行模型
    model(input_toks['input_ids'])

    # 获取真实 tokens
    real_tokens = model.tokenizer.convert_ids_to_tokens(input_toks['input_ids'][0])
    print(f"real tokens: {real_tokens}")
    real_tokens = [tok.replace('Ġ', '') for tok in real_tokens]  # 移除 'Ġ'

    # 移除 <|endoftext|> 标志
    if "<|endoftext|>" in real_tokens:
        start_idx = 1 if real_tokens[0] == "<|endoftext|>" else 0
        end_idx = -1 if real_tokens[-1] == "<|endoftext|>" else len(real_tokens)
        real_tokens = real_tokens[start_idx:end_idx]
        raw_attention_scores = raw_attention_scores[:, start_idx:end_idx, start_idx:end_idx]

    filtered_indices = [i for i, tok in enumerate(real_tokens) if tok != '']
    real_tokens = [real_tokens[i] for i in filtered_indices]
    raw_attention_scores = raw_attention_scores[:, filtered_indices, :][:, :, filtered_indices]
    print(f"Filtered real tokens: {real_tokens}")
    print(f"Raw attention scores shape for Layer {layer}, Head {head}: {raw_attention_scores.shape}")

    # 返回未经过 Softmax 的注意力分数和真实 tokens
    return raw_attention_scores[0], real_tokens  # 返回第一个 batch 的结果

def visualize_attention(attention_scores, real_tokens, layer, head, outputfile):
    """
    使用imshow可视化累积注意力分数。
    """

    # 可视化累积注意力分数
    fig = imshow(
        attention_scores.detach().cpu().numpy(),
        x=[f"{i+1}:{tok}" for i, tok in enumerate(real_tokens)],  # 带位置前缀的tokens
        y=[f"{i+1}:{tok}" for i, tok in enumerate(real_tokens)],  # 同样的tokens作为y轴
        labels={
            "x": "Keys (position:token)",
            "y": "Queries (position:token)",
            "color": " Attention Score"
        },
        title=f"Attention Scores - Layer {layer}, Head {head}",
        width=800 + len(real_tokens)*10,  # 根据序列长度动态调整宽度
        height=800,
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
    fig.write_image(outputfile, scale=2)  # 保存为PNG格式
    print(f"Saved attention visualization for Layer {layer}, Head {head} to {outputfile}")


def save_attention_data(attention_scores, real_tokens, layer, head, output_file, key):
    """
    将注意力分数保存到文件中。
    """
    # 将注意力分数转换为可序列化的格式
    attention_data = {
        "layer": layer,
        "head": head,
        "key": key,
        "tokens": real_tokens,
        "attention_scores": attention_scores.cpu().tolist()  # 转换为列表格式
    }

    # 追加写入文件
    with open(output_file, "w") as f:
        f.write(json.dumps(attention_data) + "\n")
    print(f"Saved attention data for Layer {layer}, Head {head} to {output_file}")

def save_last_token_attention_data(attention_data_list, output_file):
    """
    将所有 sequence 的最后一个 token 的注意力分数保存到一个 JSON 文件中。
    """
    with open(output_file, "w") as f:
        json.dump(attention_data_list, f, indent=4)  # 将整个列表写入文件，格式化为 JSON
    print(f"Saved all last token attention data to {output_file}")

def run(layer, head, output_dir, sequence="", picture_mode= False, outputfile="attention_scores.jsonl"):
    # 加载模型
    model_name = 'gpt2-small'
    model = HookedTransformer.from_pretrained(model_name, device='cuda')
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    print(f"\nModel loaded: {model_name}")

    input_sequence = {
        "1": {"sentence": "When Victoria and Jane got a snack at the store, Jane decided to give it to", "io": "Victoria"},
        "2": {"sentence": "Then, Tom and James had a lot of fun at the park. Tom gave a present to", "io": "James"},
        "3": {"sentence": "While Lily and Tony were working at the bank, Lily gave a hug to", "io": "Tony"},
        "4": {"sentence": "Then, Felix and Sam had a long argument. Afterwards Sam said to", "io": "Felix"},
    } if sequence == "" else sequence
    print(f"\nInput sequence: {input_sequence}")

    os.makedirs(output_dir, exist_ok=True)
    outputfile = f"{output_dir}/{outputfile}"
    attention_data_list = []
    for key, data in input_sequence.items():
        sentence = data["sentence"]
        io = data["io"]
        formatted_sentence = f"<|endoftext|> {sentence} <|endoftext|>"
        print(f"\nProcessing {key}: {formatted_sentence}")
        # 计算注意力分数
        print(f"\nComputing attention scores for Layer {layer}, Head {head}...")
        attention_scores, real_tokens = compute_attention_scores(model, formatted_sentence, layer, head)
        print(f"Attention scores computed for Layer {layer}, Head {head}.")
        
        # 处理最后一个 token 的注意力分数
        last_token_attention_scores = attention_scores[-1]  # 获取最后一个 token 的注意力分数
        important_indices = [i for i, score in enumerate(last_token_attention_scores) if score >= 0.1]
        # 选择注意力分数大于0.1的token
        # important_indices = t.topk(last_token_attention_scores, k=2).indices.tolist()
        # 选择注意力分数最高的两个token
        ## 如果某个token在句子中的出现次数大于1，则声明important_tokens时需要加token_这个特定token在整个句子中第几次出现
        important_tokens = [
            real_tokens[i] + f"_{real_tokens[:i].count(real_tokens[i]) + 1}" if real_tokens.count(real_tokens[i]) > 1 else real_tokens[i]
            for i in important_indices
        ]
        important_scores = [round(last_token_attention_scores[i].item(), 2) for i in important_indices]
        highlighted_sentence = " ".join(
            f"<<{real_tokens[i]}>>" if i in important_indices else real_tokens[i]
            for i in range(len(real_tokens))
        )
        example_1 = f"Example {key}: {highlighted_sentence} {{{{{io}}}}}\n"
        example_2 = "Activations: " + ", ".join(
            f"(\"{token}\", {score})" for token, score in zip(important_tokens, important_scores)
        ) + "\n"
        last_token_data = {
            "layer": layer,
            "head": head,
            "key": key,
            "highlighted_sentence": highlighted_sentence,
            "indirect_object": io,
            "important_tokens": [
                {"token": token, "score": score} for token, score in zip(important_tokens, important_scores)
            ],
            "original_sentence": sentence,
            "example_sentence": example_1,
            "example_activations": example_2,
            "number_of_important_tokens": len(important_tokens),
            "attention_scores": [
                {
                    "token": real_tokens[i]+f"_{real_tokens[:i].count(real_tokens[i]) + 1}" if real_tokens.count(real_tokens[i]) > 1 else real_tokens[i],
                    "score": round(last_token_attention_scores[i].item(), 2)
                }
                for i in range(len(real_tokens))
            ]
        }

        # 将数据添加到列表中
        attention_data_list.append(last_token_data)

        
        if picture_mode:
            # 可视化注意力分数
            print(f"\nVisualizing attention scores for Layer {layer}, Head {head}...")
            output_img_file = f"{output_dir}/{key}.png"
            visualize_attention(attention_scores, real_tokens, layer, head, output_img_file)
            print(f"\nVisualization complete for Layer {layer}, Head {head}!")
    
    save_last_token_attention_data(attention_data_list, outputfile)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process attention scores for a specific layer and head.")
    parser.add_argument("--layer", type=int, required=True, help="The layer index (0-based).")
    parser.add_argument("--head", type=int, required=True, help="The head index (0-based).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output dir.")
    parser.add_argument("--sequence", type=str, default="", help="The input sequence to process.")
    parser.add_argument("--picture_mode", action="store_true", help="Whether to visualize the attention scores.")
    parser.add_argument("--outputfile", type=str, default="attention_scores.jsonl", help="The output file name.")
    args = parser.parse_args()

    run(args.layer, args.head, args.output_dir, args.sequence, args.picture_mode, args.outputfile)