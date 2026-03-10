from functools import partial
import sys
from pathlib import Path
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer
import numpy as np
import json
import time
from datetime import datetime
import csv
from tqdm import tqdm
# import copy # 不再需要 copy
import argparse
# import matplotlib.pyplot as plt # 移除绘图
import traceback

# Prefer eap-ig implementation for evaluate_circuit_ratio/f.
EAP_ROOT = os.environ.get("EAP_ROOT", "/data31/private/wangziran/eap-ig")
if Path(EAP_ROOT).exists():
    sys.path.insert(0, EAP_ROOT)

# 导入 eap 相关模块
from eap.graph import Graph, InputNode # InputNode 仍然需要
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute

original_inputnode_hash = getattr(InputNode, '__hash__', None)

def safe_inputnode_hash(self):
    try:
        # 检查 name 属性是否存在，并且原始 hash 方法可调用
        if hasattr(self, 'name') and self.name is not None and callable(original_inputnode_hash):
             return original_inputnode_hash(self)
        # 如果 name 不存在或原始 hash 不可用，则使用 id 作为哈希值
        return hash(id(self))
    except AttributeError:
        # 处理可能的 AttributeError
        return hash(id(self))

if callable(original_inputnode_hash):
    InputNode.__hash__ = safe_inputnode_hash
    print("Applied safe hash patch to InputNode.")
else:
    # 如果 InputNode 原本没有 __hash__ 方法，则添加一个基于 id 的哈希方法
    InputNode.__hash__ = lambda self: hash(id(self))
    print("InputNode had no original __hash__. Added id-based hash for compatibility.")
# --- End Global Patch ---


# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 数据集类定义 ---
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    if isinstance(labels[0], (tuple, list)):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, tokenizer: PreTrainedTokenizer | None = None):
        try:
            self.df = pd.read_csv(filepath)
            print(f"加载数据集成功: {filepath}, 共{len(self.df)}个样本")
        except FileNotFoundError:
            print(f"错误: 数据集文件未找到: {filepath}")
            raise
        except Exception as e:
            print(f"加载数据集时出错: {filepath}, 错误: {e}")
            raise

        self.mode = "label"
        self.pos_ids = None
        self.neg_ids = None

        if tokenizer is not None and "correct_token" in self.df.columns and "incorrect_token" in self.df.columns:
            self.mode = "token_pair"
            self.pos_ids = []
            self.neg_ids = []
            for _, row in self.df.iterrows():
                pos = str(row["correct_token"]).strip()
                neg = str(row["incorrect_token"]).strip()
                if pos and not pos.startswith(" "):
                    pos = " " + pos
                if neg and not neg.startswith(" "):
                    neg = " " + neg
                pos_ids = tokenizer.encode(pos, add_special_tokens=False)
                neg_ids = tokenizer.encode(neg, add_special_tokens=False)
                if len(pos_ids) != 1 or len(neg_ids) != 1:
                    raise ValueError(f"Token pair must be single-token: {pos!r}, {neg!r}")
                self.pos_ids.append(pos_ids[0])
                self.neg_ids.append(neg_ids[0])
        elif "label" not in self.df.columns:
            raise ValueError("Dataset must contain label column or correct_token/incorrect_token columns.")

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)
        return self

    def head(self, n: int):
        # 返回新实例，避免修改原始数据集
        new_df = self.df.head(n).copy()
        new_dataset = EAPDataset.__new__(EAPDataset) # 创建新实例而不调用 __init__
        new_dataset.df = new_df
        new_dataset.mode = self.mode
        new_dataset.pos_ids = self.pos_ids[:n] if self.pos_ids is not None else None
        new_dataset.neg_ids = self.neg_ids[:n] if self.neg_ids is not None else None
        print(f"创建了包含前 {n} 个样本的新数据集实例。")
        return new_dataset

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.mode == "token_pair":
            return row["clean"], row["corrupted"], (self.pos_ids[index], self.neg_ids[index])
        return row['clean'], row['corrupted'], row['label']

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)

# --- 定义度量函数 (保持不变) ---
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    # 确保 input_length 是 tensor 且在正确设备上
    if not isinstance(input_length, torch.Tensor):
        try:
            input_length = torch.tensor(input_length, device=logits.device)
        except Exception as e:
            print(f"警告：无法将 input_length 转换为 tensor: {e}. 假定所有序列长度为 logits.shape[1]")
            input_length = torch.full((batch_size,), logits.shape[1], device=logits.device)

    input_length = input_length.to(logits.device)
    # 获取 input_length 指定位置的 logits (使用 input_length - 1)
    # 检查 input_length 是否有效
    if torch.any(input_length <= 0) or torch.any(input_length > logits.shape[1]):
         print(f"警告: get_logit_positions 检测到无效的 input_length 值。Min: {input_length.min()}, Max: {input_length.max()}, Logits shape[1]: {logits.shape[1]}")
         # 可以选择修正或抛出错误，这里暂时钳制到有效范围
         input_length = torch.clamp(input_length, 1, logits.shape[1])

    logits = logits[idx, input_length - 1]
    return logits

def get_sv_agreement_metric(tokenizer: PreTrainedTokenizer):
    """创建针对主语-动词一致性的评估指标"""
    singular_tokens = torch.tensor([
        tokenizer(" is").input_ids[0], tokenizer(" has").input_ids[0],
        tokenizer(" was").input_ids[0], tokenizer("'s").input_ids[0]
    ])
    plural_tokens = torch.tensor([
        tokenizer(" are").input_ids[0], tokenizer(" have").input_ids[0],
        tokenizer(" were").input_ids[0], tokenizer("'re").input_ids[0],
        tokenizer("ve").input_ids[0]
    ])
    print(f"单数动词tokens: {singular_tokens}, 复数动词tokens: {plural_tokens}")

    def agreement_metric(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False, debug=False): # 添加 debug
        pos_logits = get_logit_positions(logits, input_length) # 使用 input_length
        probs = torch.softmax(pos_logits, dim=-1)
        batch_size = probs.shape[0]
        results = torch.zeros(batch_size, device=logits.device)
        sing_tokens = singular_tokens.to(logits.device)
        plur_tokens = plural_tokens.to(logits.device)

        for i, label in enumerate(labels):
            if label == 0:
                correct_prob = probs[i, sing_tokens].sum()
                incorrect_prob = probs[i, plur_tokens].sum()
            else:
                correct_prob = probs[i, plur_tokens].sum()
                incorrect_prob = probs[i, sing_tokens].sum()
            results[i] = correct_prob - incorrect_prob

        if debug:
             print(f"SV Agreement Results (Batch): mean={results.mean().item():.4f}, acc={(results > 0).float().mean().item():.4f}")

        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    return agreement_metric

def get_token_pair_metric():
    def token_pair_metric(
        logits: torch.Tensor,
        clean_logits: torch.Tensor,
        input_length: torch.Tensor,
        labels: torch.Tensor,
        mean: bool = True,
        loss: bool = False,
        debug: bool = False,
    ):
        pos_logits = get_logit_positions(logits, input_length)
        pos_ids = labels[:, 0].to(logits.device)
        neg_ids = labels[:, 1].to(logits.device)
        pos = pos_logits.gather(1, pos_ids.unsqueeze(1)).squeeze(1)
        neg = pos_logits.gather(1, neg_ids.unsqueeze(1)).squeeze(1)
        results = pos - neg
        if debug:
            print(f"Token-pair Results (Batch): mean={results.mean().item():.4f}")
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    return token_pair_metric

# --- 保存预测详情的函数 (保持不变，但可以考虑是否在单次运行时调用) ---
# def save_prediction_details(...) # 如果不需要详细的逐样本日志，可以注释掉或移除此函数及其调用

# --- save_graph_with_performance 函数 (从 run_single.py 借鉴并适配) ---
def save_graph_with_performance(graph, file_path, baseline, performance, method):
    """保存图结构到JSON,并添加性能信息"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    graph.to_json(file_path) # 保存图的基本结构
    
    # 读取刚保存的图数据，并添加元数据
    with open(file_path, 'r') as f:
        graph_data = json.load(f)

    if "edges" not in graph_data:
        edge_dict = {}
        for edge_name, edge in graph.edges.items():
            try:
                in_graph = bool(edge.in_graph)
            except Exception:
                in_graph = False
            if not in_graph:
                continue
            edge_dict[edge_name] = {
                "parent": edge.parent.name,
                "child": edge.child.name,
                "qkv": edge.qkv,
                "score": float(edge.score),
                "hook": edge.hook,
                "index": str(edge.index),
                "in_graph": True,
            }
        graph_data["edges"] = edge_dict
    
    performance_data = {
        "metadata": {
            "method": method,
            "original_performance": baseline if not np.isnan(baseline) else None, # 处理 NaN
            "circuit_performance": performance if not np.isnan(performance) else None, # 处理 NaN
            "nodes_count": graph.count_included_nodes(),
            "edges_count": graph.count_included_edges()
        }
    }
    # 合并元数据和图数据
    result_data = {**performance_data, "graph_data": graph_data} # 将原图数据嵌套一层
    
    with open(file_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"已保存包含性能信息的图数据到: {file_path}")


# --- 修改 run_sv_agreement_analysis 函数 ---
def run_sv_agreement_analysis(data_file, output_dir, n_nodes_value: int):
    """运行主语-动词一致性电路分析 (仅 EAP-IG)，针对单个 n_nodes 值，并返回图对象"""
    os.makedirs(output_dir, exist_ok=True)
    g_pruned_final = None # 初始化

    print("=" * 80)
    print(f"开始主语-动词一致性电路分析 (EAP-IG, n_nodes = {n_nodes_value}, 重新加载)")
    print("=" * 80)

    analysis_start_time = time.time()

    # --- 模型加载 ---
    print("正在加载模型...")
    load_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_path = "/data31/private/wangziran/eap-ig/gpt2"
    try:
        hf_model = GPT2LMHeadModel.from_pretrained(local_path)
        tokenizer = GPT2Tokenizer.from_pretrained(local_path)
        model = HookedTransformer.from_pretrained(
            "gpt2", hf_model=hf_model, tokenizer=tokenizer, device=device
        )
    except Exception as e:
        print(f"从本地路径 {local_path} 加载模型失败: {e}")
        print("尝试从 HuggingFace Hub 加载 'gpt2'...")
        model = HookedTransformer.from_pretrained('gpt2', device=device)
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
             model.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    print(f"模型加载完成，耗时: {time.time() - load_start:.2f}秒, 设备: {device}")
    # --- 结束模型加载 ---

    # --- 数据加载 ---
    print(f"正在加载数据集: {data_file}")
    ds_start = time.time()
    try:
        ds = EAPDataset(data_file, tokenizer=model.tokenizer)
        batch_size = 32
        dataloader = ds.to_dataloader(batch_size)
        print(f"数据集加载完成，耗时: {time.time() - ds_start:.2f}秒")
        print(f"批次大小: {batch_size}, 总批次数: {len(dataloader)}")
    except Exception as e:
        print(f"加载数据集或创建 DataLoader 时出错: {e}")
        print(traceback.format_exc())
        if callable(original_inputnode_hash): InputNode.__hash__ = original_inputnode_hash
        return float('nan'), float('nan'), float('nan'), float('nan'), 0, 0, None
    # --- 结束数据加载 ---

    # --- 指标配置 ---
    print("配置模型评估指标...")
    if ds.mode == "token_pair":
        agreement_metric = get_token_pair_metric()
        print("Using token-pair logit-diff metric.")
    else:
        agreement_metric = get_sv_agreement_metric(model.tokenizer)
        print("Using SV agreement metric.")
    # --- 结束指标配置 ---

    # --- 基线评估 ---
    print("\n评估原始模型性能...")
    baseline_perf = float('nan')
    baseline_acc = float('nan')
    baseline_start = time.time()
    try:
        with torch.no_grad():
            baseline_tensor = evaluate_baseline(
                model,
                dataloader,
                partial(agreement_metric, loss=False, mean=False, debug=False),
            )
            baseline_perf = baseline_tensor.mean().item()
            baseline_acc = (baseline_tensor > 0).float().mean().item()
        print(f"基线评估完成，耗时: {time.time() - baseline_start:.2f}秒")
        print(f"原始模型性能 (Mean Diff): {baseline_perf:.8f}")
        print(f"原始模型准确率: {baseline_acc:.4f}")
    except Exception as e:
        print(f"评估基线时出错: {e}")
        print(traceback.format_exc())
    # --- 结束基线评估 ---

    # --- EAP-IG 归因 ---
    print("\n" + "=" * 80)
    print("使用 EAP-IG-inputs 方法进行电路提取")
    print("=" * 80)
    print("  实例化新图...")
    g_attributed = Graph.from_model(model)
    print("  开始归因分析...")
    attr_start = time.time()
    try:
        attribute(
            model, g_attributed, dataloader,
            partial(agreement_metric, loss=True, mean=True),
            method='EAP-IG-inputs',
            ig_steps=5
        )
        print(f"  归因分析完成! 耗时: {time.time() - attr_start:.2f}秒")
    except Exception as e:
        print(f"  归因分析时出错 (n_nodes={n_nodes_value}): {e}")
        print(traceback.format_exc())
        print(f"  无法为 n_nodes={n_nodes_value} 进行评估。")
        if callable(original_inputnode_hash): InputNode.__hash__ = original_inputnode_hash
        return baseline_perf, baseline_acc, float('nan'), float('nan'), 0, 0, None
    # --- 结束 EAP-IG 归因 ---

    # --- 剪枝 ---
    print(f"  提取前 {n_nodes_value} 个最重要节点...")
    g_pruned_final = g_attributed # 先赋值
    g_pruned_final.apply_topn(n_nodes_value, True)
    num_nodes = g_pruned_final.count_included_nodes()
    num_edges = g_pruned_final.count_included_edges()
    print(f"  请求节点数: {n_nodes_value}, 实际保留节点数: {num_nodes}, 边数: {num_edges}")
    # --- 结束剪枝 ---

    # --- 评估剪枝后性能 ---
    print("  评估剪枝后性能...")
    current_perf = float('nan')
    current_acc = float('nan')
    eval_start = time.time()
    try:
        if num_nodes > 0: # 只有在有节点的情况下才评估
            with torch.no_grad():
                results_tensor = evaluate_graph(
                    model,
                    g_pruned_final,
                    dataloader,
                    partial(agreement_metric, loss=False, mean=False, debug=False)
                )
                current_perf = results_tensor.mean().item()
                current_acc = (results_tensor > 0).float().mean().item()
            print(f"  评估完成! 耗时: {time.time() - eval_start:.2f}秒")
            print(f"  n_nodes={n_nodes_value}: Performance={current_perf:.6f}, Accuracy={current_acc:.4f}")
        else:
            print(f"  n_nodes={n_nodes_value}: 没有保留节点，跳过评估。")
            current_perf = float('nan')
            current_acc = float('nan')
    except Exception as e:
        print(f"  评估 EAP-IG (n={n_nodes_value}) 时出错: {e}")
        print(traceback.format_exc())
        print(f"  n_nodes={n_nodes_value}: 评估失败")
    # --- 结束评估 ---

    print(f"--- 完成分析 n_nodes = {n_nodes_value}. 总耗时: {time.time() - analysis_start_time:.2f}秒 ---")

    if callable(original_inputnode_hash):
        InputNode.__hash__ = original_inputnode_hash

    return baseline_perf, baseline_acc, current_perf, current_acc, num_nodes, num_edges, g_pruned_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EAP-IG analysis for a single n_edge value and save the graph.")
    parser.add_argument('--data_file', type=str,
                        default="../../datasets/agr_sv_num_pp_data.csv",
                        help='Path to the dataset CSV file.')
    parser.add_argument('--output_dir', type=str,
                        default="../../results/gpt2/agr_sv_num_pp_single", # 修改默认输出目录
                        help='Directory to save results and the graph JSON file.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--n_nodes', type=int,
                       help='(Legacy) Number of top nodes to keep for the analysis.')
    group.add_argument('--n_edge', type=int,
                       help='Number of top edges to keep for the analysis.')
    args = parser.parse_args()

    overall_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # 用于文件名

    try:
        n_value = args.n_edge if args.n_edge is not None else args.n_nodes
        print(f"开始运行 EAP-IG 电路分析 (数据集: {args.data_file}, 输出: {args.output_dir}, 目标边数: {n_value})...")

        # 调用分析函数
        base_perf, base_acc, current_perf, current_acc, num_nodes, num_edges, g_pruned = run_sv_agreement_analysis(
            args.data_file, args.output_dir, n_value
        )

        print("\n" + "=" * 80)
        print("分析结果总结:")
        print("=" * 80)
        print(f"目标边数 (n_edge): {n_value}")
        print(f"实际保留节点数: {num_nodes}")
        print(f"实际保留边数: {num_edges}")
        
        base_perf_str = f"{base_perf:.6f}" if not np.isnan(base_perf) else "N/A"
        base_acc_str = f"{base_acc:.4f}" if not np.isnan(base_acc) else "N/A"
        print(f"基线性能 (Mean Diff): {base_perf_str}")
        print(f"基线准确率: {base_acc_str}")

        current_perf_str = f"{current_perf:.6f}" if not np.isnan(current_perf) else "N/A (可能由于无节点或评估错误)"
        current_acc_str = f"{current_acc:.4f}" if not np.isnan(current_acc) else "N/A (可能由于无节点或评估错误)"
        print(f"剪枝后电路性能 (Mean Diff): {current_perf_str}")
        print(f"剪枝后电路准确率: {current_acc_str}")
        print("=" * 80)

        # 保存剪枝后的图信息
        if g_pruned and num_nodes > 0:
            graph_filename = os.path.join(args.output_dir, f"graph_edge_{n_value}_{timestamp}.json")
            try:
                save_graph_with_performance(g_pruned, graph_filename, base_perf, current_perf, "EAP-IG-inputs")
            except Exception as e:
                print(f"保存图到 {graph_filename} 时发生错误: {e}")
                print(traceback.format_exc())
        elif g_pruned and num_nodes == 0:
            print(f"由于剪枝后没有保留节点 (请求 {n_value} 条边), 图未保存。")
        else:
            print("由于分析过程中未成功生成图对象 (g_pruned is None)，图未保存。")


        print(f"\n分析完成! 总耗时: {time.time() - overall_start_time:.2f}秒")

    except Exception as e:
        print(f"\n脚本主流程发生错误: {e}")
        print(traceback.format_exc())
        if callable(original_inputnode_hash):
            InputNode.__hash__ = original_inputnode_hash
            # print("\nRestored original InputNode.__hash__ method after error.")

    finally:
        if callable(original_inputnode_hash) and hasattr(InputNode, '__hash__') and InputNode.__hash__ != original_inputnode_hash:
             InputNode.__hash__ = original_inputnode_hash
             # print("\nRestored original InputNode.__hash__ method at script end.")
        print("\n脚本执行结束。")
