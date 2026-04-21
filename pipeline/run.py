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
import argparse
import matplotlib.pyplot as plt
import traceback

# Prefer eap-ig implementation for evaluate_circuit_ratio/f.
EAP_ROOT = os.environ.get("EAP_ROOT", "/home/wangziran/eap-ig")
if Path(EAP_ROOT).exists():
    sys.path.insert(0, EAP_ROOT)

# 导入 eap 相关模块
from eap.graph import Graph, InputNode, AttentionNode # 添加 AttentionNode 的导入
from eap.evaluate import evaluate_graph, evaluate_baseline, evaluate_circuit_ratio
from eap.attribute import attribute

# --- Global Patch for InputNode and AttentionNode __hash__ ---
original_inputnode_hash = getattr(InputNode, '__hash__', None)
original_attentionnode_hash = getattr(AttentionNode, '__hash__', None)

def safe_node_hash(self):
    try:
        # 检查 name 属性是否存在
        if hasattr(self, 'name') and self.name is not None:
            return hash(self.name)
        # 如果 name 不存在，则使用 id 作为哈希值
        return hash(id(self))
    except AttributeError:
        # 处理可能的 AttributeError
        return hash(id(self))

# 应用补丁到 InputNode
if callable(original_inputnode_hash):
    InputNode.__hash__ = safe_node_hash
    print("Applied safe hash patch to InputNode.")
else:
    InputNode.__hash__ = lambda self: hash(id(self))
    print("InputNode had no original __hash__. Added id-based hash for compatibility.")

# 应用补丁到 AttentionNode
if callable(original_attentionnode_hash):
    AttentionNode.__hash__ = safe_node_hash
    print("Applied safe hash patch to AttentionNode.")
else:
    AttentionNode.__hash__ = lambda self: hash(id(self))
    print("AttentionNode had no original __hash__. Added id-based hash for compatibility.")
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

# --- 保存预测详情的函数 (保持不变) ---
def save_prediction_details(model, dataloader, output_dir, file_prefix="baseline"):
    """保存模型预测的详细信息到CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f"{file_prefix}_details_{timestamp}.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    singular_tokens = [
        model.tokenizer(" is").input_ids[0], model.tokenizer(" has").input_ids[0],
        model.tokenizer(" was").input_ids[0], model.tokenizer("'s").input_ids[0]
    ]
    plural_tokens = [
        model.tokenizer(" are").input_ids[0], model.tokenizer(" have").input_ids[0],
        model.tokenizer(" were").input_ids[0], model.tokenizer("'re").input_ids[0],
        model.tokenizer("ve").input_ids[0]
    ]

    header = [
        '样本ID', '文本类型', '标签类型',
        'is_logit', 'has_logit', 'was_logit', "'s_logit",
        'are_logit', 'have_logit', 'were_logit', "'re_logit", "ve_logit",
        'is_prob', 'has_prob', 'was_prob', "'s_prob",
        'are_prob', 'have_prob', 'were_prob', "'re_prob", "ve_prob",
        '单数合计概率', '复数合计概率',
        '正确类型概率', '错误类型概率',
        '差异分数', '损失值', '输入文本'
    ]

    with open(log_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"将保存预测结果到: {log_file_path}")
    device = next(model.parameters()).device
    sample_id = 0
    all_diffs = []

    with torch.no_grad():
        for batch_idx, (clean_texts, corrupt_texts, labels) in enumerate(tqdm(dataloader, desc=f"保存{file_prefix}预测结果")):
            # --- 获取输入长度 ---
            # 重新 tokenize 以获取长度，或者修改 collate_fn 返回长度
            # 简单起见，这里重新 tokenize (效率较低)
            clean_tokenized = model.tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True)
            input_lengths = clean_tokenized['attention_mask'].sum(dim=1)
            # --- 结束获取输入长度 ---

            for i, (clean, corrupt, label) in enumerate(zip(clean_texts, corrupt_texts, labels)):
                sample_id += 1
                label_type = "单数" if label == 0 else "复数"

                # --- 处理 Clean ---
                clean_tokens = model.tokenizer(clean, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx).to(device)
                clean_logits = model(clean_tokens['input_ids'])
                # 使用当前样本的长度获取 logits
                current_length = input_lengths[i]
                if current_length > 0:
                    clean_last_logits = clean_logits[0, current_length - 1]
                else: # 处理空输入或错误情况
                    print(f"警告: 样本 {sample_id} (clean) 长度为 0 或无效，使用第一个 logit。")
                    clean_last_logits = clean_logits[0, 0]

                clean_probs = torch.softmax(clean_last_logits, dim=-1)

                clean_sing_logits = clean_last_logits[singular_tokens].tolist()
                clean_sing_probs = clean_probs[singular_tokens].tolist()
                clean_plur_logits = clean_last_logits[plural_tokens].tolist()
                clean_plur_probs = clean_probs[plural_tokens].tolist()

                clean_sing_prob_sum = sum(clean_sing_probs)
                clean_plur_prob_sum = sum(clean_plur_probs)

                if label == 0:
                    clean_correct_prob = clean_sing_prob_sum
                    clean_incorrect_prob = clean_plur_prob_sum
                else:
                    clean_correct_prob = clean_plur_prob_sum
                    clean_incorrect_prob = clean_sing_prob_sum

                clean_diff = clean_correct_prob - clean_incorrect_prob
                clean_loss = -clean_diff
                all_diffs.append(clean_diff)

                clean_row = [sample_id, 'clean', label_type]
                clean_row.extend(clean_sing_logits)
                clean_row.extend(clean_plur_logits)
                clean_row.extend(clean_sing_probs)
                clean_row.extend(clean_plur_probs)
                clean_row.extend([
                    clean_sing_prob_sum, clean_plur_prob_sum,
                    clean_correct_prob, clean_incorrect_prob,
                    clean_diff, clean_loss, clean
                ])

                # --- 处理 Corrupt ---
                corrupt_tokens = model.tokenizer(corrupt, return_tensors="pt", truncation=True, max_length=model.cfg.n_ctx).to(device)
                corrupt_logits = model(corrupt_tokens['input_ids'])
                # 假设 corrupt 和 clean 长度相同
                if current_length > 0:
                     corrupt_last_logits = corrupt_logits[0, current_length - 1]
                else:
                     print(f"警告: 样本 {sample_id} (corrupt) 长度为 0 或无效，使用第一个 logit。")
                     corrupt_last_logits = corrupt_logits[0, 0]

                corrupt_probs = torch.softmax(corrupt_last_logits, dim=-1)

                corrupt_sing_logits = corrupt_last_logits[singular_tokens].tolist()
                corrupt_sing_probs = corrupt_probs[singular_tokens].tolist()
                corrupt_plur_logits = corrupt_last_logits[plural_tokens].tolist()
                corrupt_plur_probs = corrupt_probs[plural_tokens].tolist()

                corrupt_sing_prob_sum = sum(corrupt_sing_probs)
                corrupt_plur_prob_sum = sum(corrupt_plur_probs)

                if label == 0:
                    corrupt_correct_prob = corrupt_sing_prob_sum
                    corrupt_incorrect_prob = corrupt_plur_prob_sum
                else:
                    corrupt_correct_prob = corrupt_plur_prob_sum
                    corrupt_incorrect_prob = corrupt_sing_prob_sum

                corrupt_diff = corrupt_correct_prob - corrupt_incorrect_prob
                corrupt_loss = -corrupt_diff

                corrupt_row = [sample_id, 'corrupt', label_type]
                corrupt_row.extend(corrupt_sing_logits)
                corrupt_row.extend(corrupt_plur_logits)
                corrupt_row.extend(corrupt_sing_probs)
                corrupt_row.extend(corrupt_plur_probs)
                corrupt_row.extend([
                    corrupt_sing_prob_sum, corrupt_plur_prob_sum,
                    corrupt_correct_prob, corrupt_incorrect_prob,
                    corrupt_diff, corrupt_loss, corrupt
                ])

                with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(clean_row[:len(header)])
                    writer.writerow(corrupt_row[:len(header)])

    all_diffs_tensor = torch.tensor(all_diffs)
    mean_diff = all_diffs_tensor.mean().item()
    accuracy = (all_diffs_tensor > 0).float().mean().item()

    with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['性能统计'] + [''] * (len(header) - 1))
        writer.writerow(['样本总数(clean)', len(all_diffs)] + [''] * (len(header) - 2))
        writer.writerow(['平均差异(clean)', mean_diff] + [''] * (len(header) - 2))
        writer.writerow(['准确率(clean)', accuracy] + [''] * (len(header) - 2))
        # ... (其他统计信息) ...

    print(f"样本详情已保存到: {log_file_path}")
    print(f"平均性能 (Mean Diff): {mean_diff:.6f}")
    print(f"准确率: {accuracy:.4f}")

    return mean_diff, accuracy, log_file_path

# --- 修改 run_sv_agreement_analysis 函数 ---
def run_sv_agreement_analysis(data_file, output_dir, n_edge_value: int): # 修改为接收单个 n_edge 值
    """运行主语-动词一致性电路分析 (仅 EAP-IG)，针对单个 n_edge 值，每次重新加载"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"开始主语-动词一致性电路分析 (EAP-IG, n_edge = {n_edge_value}, 重新加载)")
    print("=" * 80)

    analysis_start_time = time.time()

    # --- 模型加载 ---
    print("正在加载模型...")
    load_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_path = "/home/wangziran/gpt2"
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
        # 在错误时恢复 hash patch 并返回错误指示
        if callable(original_inputnode_hash): InputNode.__hash__ = original_inputnode_hash
        return float('nan'), float('nan'), float('nan'), float('nan'), 0, 0 # 返回 NaN 和 0
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
    baseline_perf = float('nan') # 初始化为 NaN
    baseline_acc = float('nan') # 初始化为 NaN
    baseline_start = time.time()
    try:
        with torch.no_grad():
            # 传递 input_length 给 metric
            baseline_tensor = evaluate_baseline(
                model,
                dataloader,
                partial(agreement_metric, loss=False, mean=False, debug=False),
                # 需要确保 evaluate_baseline 内部能获取或传递 input_length
                # 假设 evaluate_baseline 会处理 tokenize 和长度获取
            )
            baseline_perf = baseline_tensor.mean().item()
            baseline_acc = (baseline_tensor > 0).float().mean().item()
        print(f"基线评估完成，耗时: {time.time() - baseline_start:.2f}秒")
        print(f"原始模型性能 (Mean Diff): {baseline_perf:.8f}")
        print(f"原始模型准确率: {baseline_acc:.4f}")
    except Exception as e:
        print(f"评估基线时出错: {e}")
        print(traceback.format_exc())
        # 基线评估失败，但仍可尝试后续步骤
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
        print(f"  归因分析时出错 (n_edge={n_edge_value}): {e}")
        print(traceback.format_exc())
        print(f"  无法为 n_edge={n_edge_value} 进行评估。")
        # 在错误时恢复 hash patch 并返回错误指示
        if callable(original_inputnode_hash): InputNode.__hash__ = original_inputnode_hash
        return baseline_perf, baseline_acc, float('nan'), float('nan'), 0, 0 # 返回基线和 NaN
    # --- 结束 EAP-IG 归因 ---

    # --- 剪枝 ---
    print(f"  提取前 {n_edge_value} 个最重要节点...")
    g_pruned = g_attributed # 直接使用归因后的图进行剪枝
    g_pruned.apply_topn(n_edge_value, True)
    num_nodes = g_pruned.count_included_nodes()
    num_edges = g_pruned.count_included_edges()
    print(f"  请求节点数: {n_edge_value}, 实际保留节点数: {num_nodes}, 边数: {num_edges}")
    # --- 结束剪枝 ---

    # --- 评估剪枝后性能 ---
    print("  评估剪枝后性能...")
    current_perf = float('nan')
    current_acc = float('nan')
    circuit_metrics = {}  # 存储电路评估指标
    eval_start = time.time()
    
    try:
        with torch.no_grad():
            # 评估电路性能
            results_tensor = evaluate_graph(
                model,
                g_pruned,
                dataloader,
                partial(agreement_metric, loss=False, mean=False, debug=False),
                quiet=True
            )
            current_perf = results_tensor.mean().item()
            current_acc = (results_tensor > 0).float().mean().item()
            
            # 计算电路比率指标
            print("  计算电路比率指标 (m_O, m_C, m_N, f)...")
            circuit_metrics = evaluate_circuit_ratio(
                model, g_pruned, dataloader,
                partial(agreement_metric, loss=False, mean=True, debug=False),
                quiet=True,
                intervention='patching',
                skip_clean=False
            )
            
        print(f"  评估完成! 耗时: {time.time() - eval_start:.2f}秒")
        print(f"  n_edge={n_edge_value}: Performance={current_perf:.6f}, Accuracy={current_acc:.4f}")
        
        # 打印评估指标
        if circuit_metrics:
            # 将张量转换为浮点数再格式化
            m_O = circuit_metrics.get('m_O', 'N/A')
            m_C = circuit_metrics.get('m_C', 'N/A')
            m_N = circuit_metrics.get('m_N', 'N/A')
            f = circuit_metrics.get('f', 'N/A')
            
            # 处理多元素张量
            m_O_str = f"{m_O.mean().item():.4f}" if isinstance(m_O, torch.Tensor) else "N/A"
            m_C_str = f"{m_C.mean().item():.4f}" if isinstance(m_C, torch.Tensor) else "N/A"
            m_N_str = f"{m_N.mean().item():.4f}" if isinstance(m_N, torch.Tensor) else "N/A"
            if isinstance(f, torch.Tensor):
                f_str = f"{f.mean().item():.4f}"
            elif isinstance(f, (float, int)):
                f_str = f"{f:.4f}"  # 直接格式化浮点数
            else:
                f_str = "N/A"
            
            print(f"  评估指标: m_O={m_O_str}, m_C={m_C_str}, m_N={m_N_str}, f={f_str}")
        
    except Exception as e:
        print(f"  评估 EAP-IG (n={n_edge_value}) 时出错: {e}")
        print(traceback.format_exc())
        print(f"  n_edge={n_edge_value}: 评估失败")
        circuit_metrics = {}
    
    print(f"--- 完成分析 n_edge = {n_edge_value}. 总耗时: {time.time() - analysis_start_time:.2f}秒 ---")
    
    # --- Restore original hash methods ---
    if callable(original_inputnode_hash):
        InputNode.__hash__ = original_inputnode_hash
        print("\nRestored original InputNode.__hash__ method.")
    if callable(original_attentionnode_hash):
        AttentionNode.__hash__ = original_attentionnode_hash
        print("\nRestored original AttentionNode.__hash__ method.")
    # --- End Restore ---
    
    # 获取电路指标值 (如果有)
    m_O_val = circuit_metrics.get('m_O', float('nan')).mean().item() if isinstance(circuit_metrics.get('m_O'), torch.Tensor) else float('nan')
    m_C_val = circuit_metrics.get('m_C', float('nan')).mean().item() if isinstance(circuit_metrics.get('m_C'), torch.Tensor) else float('nan')
    m_N_val = circuit_metrics.get('m_N', float('nan')).mean().item() if isinstance(circuit_metrics.get('m_N'), torch.Tensor) else float('nan')
    f = circuit_metrics.get('f', float('nan'))
    if isinstance(f, torch.Tensor):
        f_val = f.mean().item()
    elif isinstance(f, (float, int)):
        f_val = f  # 直接使用浮点数值
    else:
        f_val = float('nan')
    
    # 返回本次运行的结果，包括电路指标
    return baseline_perf, baseline_acc, current_perf, current_acc, num_nodes, num_edges, m_O_val, m_C_val, m_N_val, f_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EAP-IG analysis for Subject-Verb agreement with varying number of nodes, reloading each time.")
    parser.add_argument('--data_file', type=str,
                        default="../../datasets/agr_sv_num_pp_data.csv",
                        help='Path to the dataset CSV file.')
    parser.add_argument('--output_dir', type=str,
                        default="../../results/gpt2/agr_sv_num_pp",
                        help='Directory to save results and plots.')
    parser.add_argument('--n_edge', type=int, nargs='+',
                        default=[2560, 2300, 2100, 1900, 1700, 1500, 1300, 1100, 900, 700, 500],
                        help='List of node counts (top N) to evaluate.')
    args = parser.parse_args()

    n_edge_to_run = sorted([int(n) for n in args.n_edge], reverse=True) # 确保降序

    # --- 存储结果的列表 ---
    baseline_performances = []
    baseline_accuracies = []
    performance_results = []
    accuracy_results = []
    node_counts = []
    edge_counts = []
    m_O_values = []
    m_C_values = []
    m_N_values = []
    f_values = []
    
    overall_start_time = time.time()
    actual_n_edge_values_for_plot = [] # 存储成功运行的 n_edge 值

    overall_start_time = time.time()

    try:
        print(f"开始运行 EAP-IG 主语动词一致性电路分析 (数据集: {args.data_file}, 输出: {args.output_dir}, 节点数: {n_edge_to_run}, 每次重新加载)...")

        # --- 循环调用分析函数 ---
        for n_edge_value in tqdm(n_edge_to_run, desc="运行不同 n_edge"):
            try:
                # 调用分析函数，传入单个 n_edge 值
                base_perf, base_acc, current_perf, current_acc, num_nodes, num_edges, m_O, m_C, m_N, f = run_sv_agreement_analysis(
                    args.data_file, args.output_dir, n_edge_value
                )

                # 存储结果 (即使部分失败也存储，以便后续处理)
                baseline_performances.append(base_perf)
                baseline_accuracies.append(base_acc)
                performance_results.append(current_perf)
                accuracy_results.append(current_acc)
                node_counts.append(num_nodes)
                edge_counts.append(num_edges)
                m_O_values.append(m_O)
                m_C_values.append(m_C)
                m_N_values.append(m_N)
                f_values.append(f)
                actual_n_edge_values_for_plot.append(n_edge_value) # 记录尝试的 n_edge

            except Exception as inner_e:
                print(f"\n处理 n_edge={n_edge_value} 时发生未捕获的顶层错误: {inner_e}")
                print(traceback.format_exc())
                print(f"跳过 n_edge={n_edge_value} 并继续...")
                # 添加占位符结果
                m_O_values.append(float('nan'))
                m_C_values.append(float('nan'))
                m_N_values.append(float('nan'))
                f_values.append(float('nan'))
                baseline_performances.append(float('nan'))
                baseline_accuracies.append(float('nan'))
                performance_results.append(float('nan'))
                accuracy_results.append(float('nan'))
                node_counts.append(0)
                edge_counts.append(0)
                actual_n_edge_values_for_plot.append(n_edge_value) 
                continue # 继续下一个 n_edge
        # --- 结束循环 ---

        print(f"\n所有节点分析尝试完毕! 总耗时: {time.time() - overall_start_time:.2f}秒")

    except Exception as e:
        print(f"\n脚本主流程发生错误: {e}")
        print(traceback.format_exc())
        if callable(original_inputnode_hash):
            InputNode.__hash__ = original_inputnode_hash
            print("\nRestored original InputNode.__hash__ method after error.")

    finally:
        if callable(original_inputnode_hash) and hasattr(InputNode, '__hash__') and InputNode.__hash__ != original_inputnode_hash:
             InputNode.__hash__ = original_inputnode_hash
             print("\nRestored original InputNode.__hash__ method at script end.")
        print("\n脚本执行结束。")
