import os
import json
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from functools import partial
sys.path.append('/data63/private/chensiyuan/EAP-IG')
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
sys.path.append('../')
from eap.graph import Graph
from eap.evaluate import evaluate_graph
from eap.attribute import attribute
from extract_circuit_head import extract_head

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        clean = row['clean']
        corrupted = row['corrupted_hard']
        labels = [row['correct_idx'], row['incorrect_idx']]
    
        #print(f"Index: {index}, Clean: {clean}, Corrupted: {corrupted}, Labels: {labels}")
        return clean, corrupted, labels

    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    if logits is None:
        raise ValueError("Logits is None. Check model output.")
    if input_length is None:
        raise ValueError("Input length is None. Check tokenization.")
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def kl_div(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    if clean_logits is None:
        # 如果 clean_logits 为 None，则返回 0
        return torch.tensor(0.0, device=logits.device)
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)
    
    # 计算 softmax 概率分布
    probs = torch.nn.functional.softmax(logits, dim=-1)
    clean_probs = torch.nn.functional.softmax(clean_logits, dim=-1)
    
    # 计算 KL 散度
    kl_divergence = torch.nn.functional.kl_div(
        torch.log(probs + 1e-10),  # 避免 log(0)
        clean_probs,
        reduction='none'
    ).sum(dim=-1)
    
    if loss:
        kl_divergence = -kl_divergence
    if mean:
        kl_divergence = kl_divergence.mean()
    return kl_divergence

def filter_edges(graph, threshold=0.03):
     # 分别匹配 "aX.hY->logits" 和 "aX->aY.hZ" 格式的边（这里我们对 logits 的情况单独处理）
    pattern_logits = r"^(a\d+)(?:\.h(\d+))?->logits$"
    pattern_head2head = r"^(a\d+)(?:\.h(\d+))?->(a\d+)\.h(\d+)<([qkv])>$"
    
    # 遍历所有边
    for key, edge in graph.edges.items():
        # 只处理 in_graph 为 True 的边
        if not edge.in_graph:
            continue

        m_logits = re.match(pattern_logits, key)
        if m_logits:
            # 匹配到 sender->logits 边
            sender_node = m_logits.group(1)  # 如 "a7"
            sender_head_num = m_logits.group(2)  # 可能为空
            if sender_head_num is None:
                sender_head_num = "0"
            sender_param = f"{sender_node[1:]}.{sender_head_num}"
            # 调用 path_patching_head_to_logits 模块，传入 sender_param
            sys.argv = [
                "path_patching_head_to_logits.py",
                "--sender_head", sender_param
            ]
            from tests.experiments.path_patching_head_to_logits import main as path_patch_logits_main
            result = path_patch_logits_main()
            # 设定指标阈值，例如绝对值小于 threshold 时不保留该边
            if abs(float(result)) < threshold:
                edge.in_graph = False
            continue  # 完成该边处理后继续下一个

        m = re.match(pattern_head2head, key)
        if not m:
            continue  # 不符合格式跳过

        # 处理 a->a 型边（原有处理逻辑）
        sender_node = m.group(1)
        sender_head_num = m.group(2)
        receiver_node = m.group(3)
        receiver_head_num = m.group(4)
        receiver_input = m.group(5)

        if not (sender_node.startswith("a") and receiver_node.startswith("a")):
            continue

        if sender_head_num is None:
            sender_head_num = "0"

        sender_param = f"{sender_node[1:]}.{sender_head_num}"
        receiver_param = f"{receiver_node[1:]}.{receiver_head_num}"

        sys.argv = [
            "path_patching_head_to_head.py",
            "--sender_head", sender_param,
            "--receiver_head", receiver_param,
            "--receiver_input", receiver_input
        ]
        from tests.experiments.path_patching_head_to_head import main as path_patch_head_main
        result = path_patch_head_main()
        # 例如当指标的绝对值小于 threshold，则标记该边不在图中
        if abs(float(result)) < threshold:
            edge.in_graph = False

    # 保存新图的 JSON 文件
    graph.to_json('results/ioi/circuits/filtered_graph.json')
    # 生成新的可视化图片
    graph.to_graphviz('results/ioi/circuits/filtered_graph.png')
def main(topn): 
    model_name = 'gpt2-small'
    model = HookedTransformer.from_pretrained(model_name, device='cuda')
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    ds = EAPDataset('ioi_llama.csv')
    dataloader = ds.to_dataloader(10)

    g = Graph.from_model(model)
    # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
    attribute(model, g, dataloader, partial(kl_div, loss=True, mean=True), method='EAP-IG-inputs',ig_steps=5)

    # We can now apply greedy search to the scored graph to find a circuit! We prune dead nodes, and export the circuit.

    g.apply_topn(topn, True)
    g.to_json(f'results/ioi/circuits/graph.json')
    filter_edges(g, threshold=0.03)
    g.to_json(f'results/ioi/circuits/ultimate_graph.json')
    ultimate_json_data = json.load(open('results/ioi/circuits/ultimate_graph.json'))
    extract_head(ultimate_json_data, f'topn={topn}')
    edges = g.count_included_edges()
    # We can then convert our circuit into a visualization!

    gz = g.to_graphviz(f'results/ioi/circuits/graph.png')

    # We then evaluate our model's metric score as opposed to a baseline.
    results = evaluate_graph(model, g, dataloader, partial(kl_div, loss=False, mean=True),skip_clean=False).mean().item()
    result_data = {
        'topn': topn,
        'edges': edges,
        'results': results
    }
    file_path = 'results/ioi/circuits/result_ioi_topn.json'
    # 如果文件存在，读取现有数据
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                existing_data = json.load(f)  # 读取现有 JSON 列表
            except json.JSONDecodeError:
                # 如果文件内容不是有效的 JSON，初始化一个空列表
                existing_data = []
    else:
        # 如果文件不存在，初始化一个空列表
        existing_data = []

    # 将新数据追加到列表中
    existing_data.append(result_data)

    # 将更新后的列表写入文件
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)  # indent=4 用于美化输出

    print(f"TopN: {topn}, Edges: {edges}, Results: {results}")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ioi.py <topn>")
        sys.exit(1)
    topn = float(sys.argv[1])
    main(int(topn))
