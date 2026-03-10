import torch
import os
import pandas as pd
import json
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
from eap.graph import Graph
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline
from tqdm import tqdm
import time

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class EAPDataset(Dataset):
    def __init__(self, filepath_or_df):
        if isinstance(filepath_or_df, str):
            self.df = pd.read_csv(filepath_or_df)
        else:
            self.df = filepath_or_df

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], row['label']

    def to_dataloader(self, batch_size: int):
        def collate_EAP(xs):
            clean, corrupted, labels = zip(*xs)
            clean = list(clean)
            corrupted = list(corrupted)
            return clean, corrupted, labels

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

def gender_prob_diff(tokenizer):
    """计算性别代词概率差异的指标函数"""
    he_token = tokenizer(" he").input_ids[0]
    she_token = tokenizer(" she").input_ids[0]

    def prob_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        # 获取预测位置的logits
        batch_size = logits.size(0)
        
        idx = torch.arange(batch_size, device=logits.device)
        pos_logits = logits[idx, input_length - 1]

        # 打印前10个样本的前10个候选词
        print_samples = min(10, batch_size)
        print("\n===== 目标位置前十候选词 =====")
        for i in range(print_samples):
            # 获取当前样本的预测分数
            sample_logits = pos_logits[i]
            # 计算概率
            probs = torch.softmax(sample_logits, dim=-1)
            # 获取前10个最可能的token
            topk_values, topk_indices = torch.topk(probs, 10)
            
            # 解码token为文字
            topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
            
            print(f"样本 {i+1} (标签: {labels[i]}), 预期: {'he' if labels[i] == 0 else 'she'}")
            for j, (token, prob, idx) in enumerate(zip(topk_tokens, topk_values, topk_indices)):
                # 特别标记he和she
                marker = ""
                if idx.item() == he_token:
                    marker = " <-- HE"
                elif idx.item() == she_token:
                    marker = " <-- SHE"
                print(f"  {j+1}. '{token}' (概率: {prob:.4f}, ID: {idx.item()}){marker}")
            print("-" * 40)
        print("===== 候选词打印完成 =====\n")

        # 计算he和she的概率
        probs = torch.softmax(pos_logits, dim=-1)
        he_probs = probs[:, he_token]
        she_probs = probs[:, she_token]

        # 根据标签计算概率差异
        results = []
        for i, label in enumerate(labels):
            if label == 0:  # 应该是"he"
                results.append(he_probs[i] - she_probs[i])
            else:  # 应该是"she"
                results.append(she_probs[i] - he_probs[i])

        # 使用stack保留梯度
        results = torch.stack(results)
        
        if loss:
            results = -results  # 转为损失函数形式
        if mean:
            results = results.mean()
        return results

    return prob_diff

def save_graph_with_performance(graph, file_path, baseline, performance, method):
    """保存图结构到JSON,并添加性能信息"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 首先生成JSON文件
    graph.to_json(file_path)

    # 重新读取JSON文件
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    # 添加性能信息
    performance_data = {
        "metadata": {
            "method": method,
            "original_performance": baseline,
            "circuit_performance": performance,
            "nodes_count": graph.count_included_nodes(),
            "edges_count": graph.count_included_edges()
        }
    }

    # 合并数据
    result_data = {**performance_data, **graph_data}

    # 重新保存到文件，添加缩进格式化
    with open(file_path, 'w') as f:
        json.dump(result_data, f, indent=2)

    # 生成图像
    graph.to_graphviz(file_path.replace('.json', '.png'))

    print(f"已保存图数据到: {file_path}")

def run_eap_analysis(data_file, output_dir="./results/gpt2/gender_graph"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("开始性别一致性电路分析")
    print("=" * 80)

    # 记录开始时间
    start_time = time.time()
    print("正在加载模型")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained('gpt2', device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    print("模型加载完成")
    
    # 加载数据集
    load_start = time.time()
    print(f"正在加载数据集: {data_file}")
    ds = EAPDataset(data_file)
    print(f"共加载了 {len(ds)} 个样本, 耗时: {time.time() - load_start:.2f}秒")

    batch_size = 32
    dataloader = ds.to_dataloader(batch_size)
    print(f"批次大小: {batch_size}, 总批次数: {len(dataloader)}")

    # 定义指标
    print("配置模型评估指标...")
    prob_diff = gender_prob_diff(model.tokenizer)

    # 评估原始模型性能（基线）
    print("评估原始模型性能...")
    with torch.no_grad():
        baseline = evaluate_baseline(model, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"原始模型性能: {baseline:.8f}")

    # ============ EAP-IG 方法 ============
    print("\n" + "=" * 80)
    print("使用 EAP-IG-inputs 方法进行电路提取")
    print("=" * 80)
    
    # 实例化图
    g = Graph.from_model(model)
    
    # 使用EAP-IG进行归因
    attr_start = time.time()
    attribute(
        model, g, dataloader,
        partial(prob_diff, loss=True, mean=True),
        method='EAP-IG-inputs',
        ig_steps=5
    )
    print(f"归因分析完成! 耗时: {time.time() - attr_start:.2f}秒")
    
    # 保留前N个节点
    n_nodes = 200
    print(f"提取前 {n_nodes} 个最重要节点...")
    g.apply_topn(n_nodes, True)
    
    # 评估剪枝后的性能
    with torch.no_grad():
        results = evaluate_graph(model, g, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"原始性能为 {baseline:.8f}; EAP-IG-inputs 电路性能为 {results:.8f}")
    print(f"电路节点数: {g.count_included_nodes()}, 边数: {g.count_included_edges()}")
    
    # 保存EAP-IG电路
    eap_ig_file = os.path.join(output_dir, "gender_graph_eap_ig.json")
    save_graph_with_performance(g, eap_ig_file, baseline, results, "EAP-IG-inputs")
'''
    # ============ 普通EAP方法 ============
    print("\n" + "=" * 80)
    print("使用 EAP 方法进行电路提取")
    print("=" * 80)
    
    g_eap = Graph.from_model(model)
    
    attr_start = time.time()
    attribute(
        model, g_eap, dataloader,
        partial(prob_diff, loss=True, mean=True),
        method='EAP'
    )
    print(f"归因分析完成! 耗时: {time.time() - attr_start:.2f}秒")
    
    g_eap.apply_topn(n_nodes, True)
    
    with torch.no_grad():
        results_eap = evaluate_graph(model, g_eap, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"原始性能为 {baseline:.4f}; EAP 电路性能为 {results_eap:.4f}")
    print(f"电路节点数: {g_eap.count_included_nodes()}, 边数: {g_eap.count_included_edges()}")
    
    # 保存EAP电路
    eap_file = os.path.join(output_dir, "gender_graph_eap.json")
    save_graph_with_performance(g_eap, eap_file, baseline, results_eap, "EAP")

    # ============ Clean-Corrupted方法 ============
    print("\n" + "=" * 80)
    print("使用 clean-corrupted 方法进行电路提取")
    print("=" * 80)
    
    g_cc = Graph.from_model(model)
    
    attr_start = time.time()
    attribute(
        model, g_cc, dataloader,
        partial(prob_diff, loss=True, mean=True),
        method='clean-corrupted'
    )
    print(f"归因分析完成! 耗时: {time.time() - attr_start:.2f}秒")
    
    g_cc.apply_topn(n_nodes, True)
    
    with torch.no_grad():
        results_cc = evaluate_graph(model, g_cc, dataloader, partial(prob_diff, loss=False, mean=False)).mean().item()
    print(f"原始性能为 {baseline:.4f}; clean-corrupted 电路性能为 {results_cc:.4f}")
    print(f"电路节点数: {g_cc.count_included_nodes()}, 边数: {g_cc.count_included_edges()}")
    
    # 保存clean-corrupted电路
    cc_file = os.path.join(output_dir, "gender_graph_cc.json")
    save_graph_with_performance(g_cc, cc_file, baseline, results_cc, "clean-corrupted")

    # 打印结果比较
    print("\n" + "=" * 80)
    print("三种方法性能比较:")
    print(f"原始模型性能: {baseline:.4f}")
    print(f"EAP-IG-inputs 电路性能: {results:.4f}")
    print(f"EAP 电路性能: {results_eap:.4f}")
    print(f"clean-corrupted 电路性能: {results_cc:.4f}")
    print("=" * 80)

    # 计算总耗时
    total_time = time.time() - start_time
    print(f"总耗时: {total_time:.2f}秒")
    
    return g, g_eap, g_cc

if __name__ == "__main__":
    try:
        # 使用新的数据集路径
        dataset_path = "./datasets/agr_gender_eap_data.csv"
        output_dir = "./results/gpt2/gender_graph"
        
        print("开始运行EAP-IG电路分析...")
        g = run_eap_analysis(dataset_path, output_dir)

        print("\n全部完成!")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        print(traceback.format_exc())
'''