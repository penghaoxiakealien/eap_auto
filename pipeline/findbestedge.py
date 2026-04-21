import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import pandas as pd

# 确保 TOKENIZERS_PARALLELISM 环境变量在早期设置，以避免 HuggingFace Tokenizers 的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_with_n_edges(n_edges_value, data_file, output_dir, script_path=None):
    """使用指定的n_edges_value运行run.py脚本并返回结果"""
    if script_path is None:
        # 默认使用与本脚本同目录下的run.py
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    
    print(f"\n{'='*80}")
    print(f"运行 n_edge={n_edges_value} (脚本: {script_path})...")
    
    cmd = [
        sys.executable, # 使用 sys.executable 保证使用当前 Python 解释器
        script_path,
        "--data_file", data_file,
        "--output_dir", output_dir,
        "--n_edge", str(n_edges_value)
    ]
    
    try:
        start_time = time.time()
        # 使用 Popen 以便更好地控制和捕获输出
        env = os.environ.copy()
        env.setdefault("EAP_ROOT", "/home/wangziran/eap-ig")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Python 3.7+
            bufsize=1,  # 行缓冲
            universal_newlines=True,  # 确保跨平台换行符一致性
            env=env,
        )
        
        stdout_lines = []
        stderr_lines = []

        print(f"--- run.py (n_edge={n_edges_value}) STDOUT ---")
        for line in process.stdout:
            print(line.strip())
            stdout_lines.append(line)
        
        print(f"--- run.py (n_edge={n_edges_value}) STDERR ---")
        for line in process.stderr:
            print(f"STDERR: {line.strip()}")
            stderr_lines.append(line)
            
        return_code = process.wait()
        elapsed = time.time() - start_time
        
        # 收集结果
        # 传递 stdout 和 stderr 以便更全面地解析或调试
        result = parse_new_run_output(''.join(stdout_lines), ''.join(stderr_lines))
        result['n_edge'] = n_edges_value
        result['elapsed'] = elapsed
        result['success'] = (return_code == 0)
        
        if return_code != 0:
            print(f"警告: run.py (n_edge={n_edges_value}) 返回非零退出码: {return_code}")

        print(f"n_edge={n_edges_value} 运行{'成功' if result['success'] else '失败'}, 耗时: {elapsed:.2f}秒")
        print(f"解析结果: 准确率={result.get('accuracy', 'N/A')}, f值={result.get('f', 'N/A')}")
        return result
        
    except Exception as e:
        print(f"运行 n_edge={n_edges_value} 时发生严重错误: {e}")
        import traceback
        print(traceback.format_exc())
        return {'n_edge': n_edges_value, 'success': False, 'accuracy': float('nan'), 'performance': float('nan'), 'f': float('nan')}

def parse_new_run_output(output_text, error_text):
    """从new_run.py的输出文本中解析关键结果"""
    result = {
        'performance': float('nan'),
        'accuracy': float('nan'),
        'm_O': float('nan'),
        'm_C': float('nan'),
        'm_N': float('nan'),
        'f': float('nan'),
        'baseline_performance': float('nan'),
        'baseline_accuracy': float('nan'),
        'actual_nodes': 0,
        'actual_edges': 0
    }
    
    for line in output_text.splitlines():
        # 查找 "n_edge=X: Performance=Y, Accuracy=Z" 格式的行
        if f"n_edge=" in line and "Performance=" in line and "Accuracy=" in line:
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                if "Performance=" in part:
                    try:
                        result['performance'] = float(part.split("=")[1])
                    except ValueError:
                        pass # 保持 NaN
                elif "Accuracy=" in part:
                    try:
                        result['accuracy'] = float(part.split("=")[1])
                    except ValueError:
                        pass
        
        # 查找 "评估指标: m_O=X, m_C=Y, m_N=Z, f=W" 格式的行
        if "评估指标:" in line:
            metrics_str = line.split("评估指标:")[1].strip()
            metrics_parts = metrics_str.split(',')
            for m_part in metrics_parts:
                key_val = m_part.split('=')
                if len(key_val) == 2:
                    key = key_val[0].strip()
                    val_str = key_val[1].strip()
                    try:
                        result[key] = float(val_str) if val_str != "N/A" else float('nan')
                    except ValueError:
                        pass # 保持 NaN
        
        # 查找基线性能
        if "原始模型性能 (Mean Prob Diff):" in line:
            try:
                result['baseline_performance'] = float(line.split(":")[1].strip())
            except ValueError:
                pass
                
        # 查找基线准确率
        if "原始模型准确率:" in line:
            try:
                result['baseline_accuracy'] = float(line.split(":")[1].strip())
            except ValueError:
                pass
                
        # 查找实际保留的节点数和边数
        if "请求节点数:" in line and "实际保留节点数:" in line and "边数:" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "实际保留节点数:" in part:
                    try:
                        result['actual_nodes'] = int(part.split(":")[1].strip())
                    except ValueError:
                        pass
                elif "边数:" in part:
                    try:
                        result['actual_edges'] = int(part.split(":")[1].strip())
                    except ValueError:
                        pass
    return result

def find_accuracy_cliff(df, threshold=0.1): # 阈值可能需要调整
    """
    找出准确率下降明显的区间
    threshold: 准确率下降超过此值视为明显下降
    返回: (upper_nodes, lower_nodes) 表示第一个准确率下降超过阈值的区间
    """
    if df.empty or len(df) < 2:
        print("数据不足，无法寻找准确率悬崖点。")
        return None, None
    
    # 按n_edge从大到小排序
    df_sorted = df.sort_values('n_edge', ascending=False).reset_index(drop=True)
    
    # 计算相邻点的准确率差异
    df_sorted['accuracy_diff'] = df_sorted['accuracy'].diff(-1) # diff(-1) 计算当前行与下一行的差
    
    cliff_points = pd.DataFrame() # 初始化
    # 确保 accuracy_diff 列存在且非空
    if 'accuracy_diff' in df_sorted.columns and not df_sorted['accuracy_diff'].isna().all():
        # accuracy_diff > threshold 意味着 accuracy[i] - accuracy[i+1] > threshold (因为 diff(-1))
        # 即从较大的 n_edge 到较小的 n_edge，准确率下降了 threshold 以上
        cliff_points = df_sorted[df_sorted['accuracy_diff'] > threshold]

    if cliff_points.empty:
        print(f"未找到准确率下降超过 {threshold} 的点。尝试寻找最大相对下降点。")
        if 'accuracy_diff' in df_sorted.columns and 'n_edge' in df_sorted.columns and not df_sorted['n_edge'].diff(-1).abs().eq(0).all():
            df_sorted['normalized_drop'] = df_sorted['accuracy_diff'].fillna(0) / df_sorted['n_edge'].diff(-1).abs().replace(0, np.nan)
            if not df_sorted['normalized_drop'].isna().all() and not df_sorted['normalized_drop'].empty:
                max_drop_idx = df_sorted['normalized_drop'].idxmax()
                if max_drop_idx < len(df_sorted) -1:
                    upper_nodes = df_sorted.loc[max_drop_idx, 'n_edge']
                    lower_nodes = df_sorted.loc[max_drop_idx + 1, 'n_edge']
                    return upper_nodes, lower_nodes
        print("无法确定规范化的下降区间。")
        return None, None # 如果还是找不到，返回None
    else:
        # 取第一个准确率下降超过阈值的点
        first_cliff_idx = cliff_points.index[0]
        upper_nodes = df_sorted.loc[first_cliff_idx, 'n_edge']
        if first_cliff_idx + 1 < len(df_sorted):
            lower_nodes = df_sorted.loc[first_cliff_idx + 1, 'n_edge']
        else: # 如果悬崖点是最后一个点，没有更低的区间
            lower_nodes = df_sorted['n_edge'].min() # 或者返回 None，或一个合理的最小值
            print(f"警告: 准确率悬崖点 ({upper_nodes}) 是数据中的最小节点数。细化区间可能受限。")
            if upper_nodes == lower_nodes and len(df_sorted) > 1: #尝试取上一个点作为下界
                 lower_nodes = df_sorted.loc[first_cliff_idx -1 , 'n_edge'] if first_cliff_idx > 0 else upper_nodes / 2


        return upper_nodes, lower_nodes
    
    return None, None


def find_optimal_edge_count_percentile(df, baseline_accuracy, accuracy_percentile=0.3, f_percentile=0.7):
    """
    使用百分位数方法找到最佳的节点数：
    1. 找出准确率下降在前accuracy_percentile百分位的所有点
    2. 在这些点中,找出f值在前f_percentile百分位的最小节点数
    """
    if df.empty or 'accuracy' not in df.columns or 'f' not in df.columns or 'n_edge' not in df.columns:
        print("数据不完整，无法寻找最优节点数。")
        return None
    
    df_valid = df.dropna(subset=['accuracy', 'f', 'n_edge']).copy()
    if df_valid.empty:
        print("筛选掉NaN后数据为空，无法寻找最优节点数。")
        return None

    # 按n_edge从大到小排序
    df_valid = df_valid.sort_values('n_edge', ascending=False).reset_index(drop=True)
    
    # 计算每个点的准确率下降
    df_valid['accuracy_drop'] = baseline_accuracy - df_valid['accuracy']
    
    # 准确率下降阈值 (较小的下降更好)
    # quantile(q) 返回第 q*100 百分位数。accuracy_percentile=0.3 意味着我们容忍的下降程度不能超过30%最差的那些点。
    # 或者说，我们希望准确率下降值小于等于第 accuracy_percentile 百分位数。
    accuracy_drop_threshold = df_valid['accuracy_drop'].quantile(accuracy_percentile)
    print(f"准确率下降阈值 (可接受的最大下降值，基于前{accuracy_percentile*100:.0f}%的保持度): {accuracy_drop_threshold:.4f}")

    acceptable_df = df_valid[df_valid['accuracy_drop'] <= accuracy_drop_threshold]
    
    if acceptable_df.empty:
        print("警告: 没有找到满足准确率下降阈值的点。尝试选择准确率下降最小的点。")
        if not df_valid.empty:
            best_accuracy_idx = df_valid['accuracy_drop'].idxmin()
            return df_valid.loc[best_accuracy_idx, 'n_edge']
        return None # 如果 df_valid 也为空
    
    # f值阈值 (较大的f值更好)
    # quantile(1 - f_percentile) 意味着我们想要f值至少是 (1-f_percentile) 百分位数那么大。
    # 例如 f_percentile=0.7, 我们想要f值至少是第30百分位数那么大。
    # 或者，如果我们想要f值在前f_percentile (e.g. top 30%, f_percentile=0.3), 那么应该是 quantile(1-0.3) = quantile(0.7)
    # 这里的 f_percentile 参数描述是 "f值目标的百分位数(0-1)"，如果目标是top 30%，则 f_percentile=0.3, 阈值是 quantile(1-0.3)
    # 如果参数描述是 "f值至少要达到的百分位"，如 f_percentile=0.7 (至少达到70%分位)，则阈值是 quantile(0.7)
    # 从 findbestedge.py 来看, f_percentile=0.7, quantile(1-0.7) = quantile(0.3) -> f值在前70%的点 (即大于30%的点)
    # 我将保持与 findbestedge.py 一致的逻辑：f_percentile=0.7 意味着我们想要 f 值至少是第 (1-0.7)=0.3 百分位数。
    # 即，我们筛选出 f 值排在前 70% 的那些点。
    f_threshold = acceptable_df['f'].quantile(1 - f_percentile) 
    print(f"f值阈值 (筛选掉后 {f_percentile*100:.0f}% 的较低f值后，f的最小可接受值): {f_threshold:.4f}")
    
    high_f_df = acceptable_df[acceptable_df['f'] >= f_threshold]
    
    if not high_f_df.empty:
        best_node_val = high_f_df['n_edge'].min() # 选择满足条件的最小节点数
        print(f"同时满足准确率和f值条件的节点数: {sorted(high_f_df['n_edge'].unique().tolist())}")
        print(f"选择其中最小的节点数: {best_node_val}")
        return best_node_val
    else:
        print(f"警告: 在满足准确率要求的点中，没有找到f值 >= {f_threshold:.4f} 的点。将从准确率可接受的点中选择f值最大的。")
        if not acceptable_df.empty:
            best_f_idx = acceptable_df['f'].idxmax()
            return acceptable_df.loc[best_f_idx, 'n_edge']
        return None # 如果 acceptable_df 也为空

def plot_results(df, output_dir, phase="initial", baseline_accuracy=None, baseline_performance=None):
    """绘制结果图表"""
    if df.empty or 'n_edge' not in df.columns:
        print("没有可用数据或缺少 'n_edge' 列进行绘图。")
        return
    
    df_plot = df.dropna(subset=['n_edge']).sort_values('n_edge', ascending=False).reset_index(drop=True)
    if df_plot.empty:
        print("筛选掉n_edge为NaN后数据为空，无法绘图。")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 第一个子图：准确率和性能
    ax1.set_title(f"准确率和性能随节点数的变化 ({phase}阶段)", fontsize=14)
    ax1.set_xlabel("节点数 (对数刻度)", fontsize=12)
    ax1.set_ylabel("指标值", fontsize=12)
    
    if 'accuracy' in df_plot.columns and not df_plot['accuracy'].isna().all():
        ax1.plot(df_plot['n_edge'], df_plot['accuracy'], 'o-', color='royalblue', label='准确率', linewidth=2, markersize=7)
    if 'performance' in df_plot.columns and not df_plot['performance'].isna().all():
        ax1.plot(df_plot['n_edge'], df_plot['performance'], 's--', color='forestgreen', label='性能 (Mean Prob Diff)', linewidth=2, markersize=7)
    
    if baseline_accuracy is not None and not np.isnan(baseline_accuracy):
        ax1.axhline(y=baseline_accuracy, color='crimson', linestyle=':', label=f'基线准确率: {baseline_accuracy:.4f}', linewidth=2)
    if baseline_performance is not None and not np.isnan(baseline_performance):
        ax1.axhline(y=baseline_performance, color='darkorange', linestyle=':', label=f'基线性能: {baseline_performance:.4f}', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.invert_xaxis() # 节点数越多越靠左
    ax1.legend(fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # 第二个子图：m_C, m_O, m_N, f
    ax2.set_title(f"电路指标随节点数的变化 ({phase}阶段)", fontsize=14)
    ax2.set_xlabel("节点数 (对数刻度)", fontsize=12)
    ax2.set_ylabel("电路指标值 (m_O, m_C, m_N)", fontsize=12, color='darkslateblue')
    
    if 'm_C' in df_plot.columns and not df_plot['m_C'].isna().all():
        ax2.plot(df_plot['n_edge'], df_plot['m_C'], 'o-', color='deepskyblue', label='m_C', linewidth=2, markersize=7)
    if 'm_O' in df_plot.columns and not df_plot['m_O'].isna().all():
        ax2.plot(df_plot['n_edge'], df_plot['m_O'], 's-', color='salmon', label='m_O', linewidth=2, markersize=7)
    if 'm_N' in df_plot.columns and not df_plot['m_N'].isna().all():
        ax2.plot(df_plot['n_edge'], df_plot['m_N'], '^-', color='mediumseagreen', label='m_N', linewidth=2, markersize=7)
    
    ax2.tick_params(axis='y', labelcolor='darkslateblue', labelsize=10)
    ax2.set_xscale('log')
    ax2.invert_xaxis()

    lines, labels = ax2.get_legend_handles_labels()
        
    if 'f' in df_plot.columns and not df_plot['f'].isna().all():
        ax3 = ax2.twinx()
        ax3.plot(df_plot['n_edge'], df_plot['f'], 'D-', color='purple', label='f值 (右轴)', linewidth=2, markersize=7)
        ax3.set_ylabel("f值", color='purple', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='purple', labelsize=10)
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
    else:
        ax2.legend(lines, labels, loc='best', fontsize=10)
        
    # 为节点数值添加标签 (避免重叠)
    # for i, n_edge_val in enumerate(df_plot['n_edge']):
    #     if 'accuracy' in df_plot.columns and not df_plot['accuracy'].isna().iloc[i]:
    #         ax1.annotate(f"{int(n_edge_val)}", (n_edge_val, df_plot['accuracy'].iloc[i]), textcoords="offset points", 
    #                     xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout(pad=2.0) # 增加一点填充
    fig.suptitle(f"Garden NPZ v-trans (mod) Circuit Search ({phase} phase)", fontsize=16, y=0.99)
    plt.subplots_adjust(top=0.93) # 为总标题留出空间

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"edge_search_{phase}_{timestamp_str}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"图表已保存到: {plot_filename}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="自动寻找最佳保留边数")
    parser.add_argument('--data_file', type=str, 
                      default="/home/wangziran/eap_auto/datasets/garden/garden_npz_v_trans_mod.csv",
                      help='数据集路径')
    parser.add_argument('--output_dir', type=str, 
                      default="/home/wangziran/eap_auto/results/garden_search",
                      help='输出目录')
    parser.add_argument('--script_path', type=str, 
                      default=None, 
                      help='run.py脚本路径,默认为与本脚本同目录')
    parser.add_argument('--f_percentile', type=float, 
                      default=0.7, # 目标f值位于前70% (即大于第30百分位数)
                      help='f值目标的百分位数(0-1), 表示筛选掉后 f_percentile 的较低f值')
    parser.add_argument('--accuracy_percentile', type=float, 
                      default=0.3, # 准确率下降值位于前30% (较小的下降)
                      help='准确率下降的百分位数(0-1), 表示可接受的最大准确率下降程度所处的百分位')
    parser.add_argument('--cliff_threshold', type=float,
                        default=0.05,
                        help='用于find_accuracy_cliff的准确率下降阈值')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"{'='*80}")
    print("开始自动寻找最佳保留边数")
    print(f"数据集: {args.data_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"被调用脚本: {args.script_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.py')}")
    print(f"准确率下降容差百分位 (目标是值小): {args.accuracy_percentile}")
    print(f"f值目标百分位 (目标是值大): {args.f_percentile}")
    print(f"{'='*80}")
    
    overall_start = time.time()
    
    # 第一阶段：粗略搜索
    initial_edge_counts = [6400, 3200, 1600, 800, 400, 200, 100]
    print(f"\n{'='*40} 第一阶段：粗略搜索 - 边数: {initial_edge_counts} {'='*40}")
    
    results_phase1 = []
    for n_edge_val in initial_edge_counts:
        result = run_with_n_edges(n_edge_val, args.data_file, args.output_dir, args.script_path)
        results_phase1.append(result) # 即使失败也添加，便于记录
    
    initial_df = pd.DataFrame([r for r in results_phase1 if r and r.get('success')]) # 只用成功的进行分析
    if initial_df.empty and results_phase1: # 如果都失败了，但有记录
        initial_df_all_attempts = pd.DataFrame(results_phase1)
        initial_df_all_attempts.to_csv(os.path.join(args.output_dir, f"phase1_all_attempts_results_{timestamp_str}.csv"), index=False)
        print(f"\n警告: 第一阶段所有运行均未成功解析结果或运行失败。结果已尝试保存。")
        # 不进行后续阶段
        print(f"总耗时: {(time.time() - overall_start) / 60:.2f}分钟")
        return
    elif initial_df.empty:
        print("\n错误: 第一阶段没有收集到任何结果。脚本终止。")
        return

    initial_df.to_csv(os.path.join(args.output_dir, f"phase1_successful_results_{timestamp_str}.csv"), index=False)
    print(f"\n第一阶段成功结果已保存。")
    
    # 从成功的结果中获取基线
    baseline_accuracy = initial_df['baseline_accuracy'].median() if 'baseline_accuracy' in initial_df.columns and not initial_df['baseline_accuracy'].isna().all() else None
    baseline_performance = initial_df['baseline_performance'].median() if 'baseline_performance' in initial_df.columns and not initial_df['baseline_performance'].isna().all() else None

    if baseline_accuracy is None or np.isnan(baseline_accuracy):
        print("警告: 未能从第一阶段结果中获取有效的基线准确率。后续最优选择可能不准确。")
        # 尝试从所有尝试中获取，以防成功的里面没有
        if results_phase1:
            temp_baselines = [r.get('baseline_accuracy') for r in results_phase1 if r and r.get('baseline_accuracy') is not None and not np.isnan(r.get('baseline_accuracy'))]
            if temp_baselines: baseline_accuracy = np.median(temp_baselines)
    if baseline_performance is None or np.isnan(baseline_performance):
         print("警告: 未能从第一阶段结果中获取有效的基线性能。")


    plot_results(initial_df, args.output_dir, "phase1", baseline_accuracy, baseline_performance)
    
    upper_nodes, lower_nodes = find_accuracy_cliff(initial_df, threshold=args.cliff_threshold)
    
    all_results_list = results_phase1 # 开始收集所有结果
    
    if upper_nodes is not None and lower_nodes is not None and upper_nodes > lower_nodes:
        print(f"\n准确率下降明显的区间 (基于阈值 {args.cliff_threshold}): {upper_nodes} -> {lower_nodes}")
        
        # 第二阶段：细粒度搜索
        num_refined_steps = 5 # 在区间内取几个点
        step_size = (upper_nodes - lower_nodes) / num_refined_steps if num_refined_steps > 0 else 0
        refined_nodes_counts = sorted(list(set([
            round(lower_nodes + i * step_size) for i in range(num_refined_steps + 1)
            if round(lower_nodes + i * step_size) > 0 # 确保边数大于0
        ])), reverse=True) # 从大到小

        # 确保包含边界，并去除与第一阶段重复的点
        refined_nodes_counts = [n for n in refined_nodes_counts if n not in initial_df['n_edge'].values]
        
        if refined_nodes_counts:
            print(f"\n{'='*40} 第二阶段：细粒度搜索 - 边数: {refined_nodes_counts} {'='*40}")
            results_phase2 = []
            for n_edge_val in refined_nodes_counts:
                result = run_with_n_edges(n_edge_val, args.data_file, args.output_dir, args.script_path)
                results_phase2.append(result)
            all_results_list.extend(results_phase2)
        else:
            print("\n第二阶段：没有新的节点数需要运行（可能已在第一阶段覆盖或区间太小）。")
    else:
        print("\n未能确定细化搜索区间，跳过第二阶段。将基于第一阶段结果寻找最优节点数。")

    all_successful_results = [r for r in all_results_list if r and r.get('success')]
    if not all_successful_results:
        print("错误: 所有阶段均未收集到成功的运行结果。无法继续。")
        if all_results_list: # 保存所有尝试，即使失败
            all_df_attempts = pd.DataFrame(all_results_list)
            all_df_attempts.to_csv(os.path.join(args.output_dir, f"all_attempts_results_{timestamp_str}.csv"), index=False)
        return

    all_df = pd.DataFrame(all_successful_results)
    all_df.to_csv(os.path.join(args.output_dir, f"all_successful_results_{timestamp_str}.csv"), index=False)
    print(f"\n所有成功运行的结果已保存。")
    
    # 更新基线值，使用所有成功运行的中位数，更稳健
    final_baseline_accuracy = all_df['baseline_accuracy'].median() if 'baseline_accuracy' in all_df.columns and not all_df['baseline_accuracy'].isna().all() else baseline_accuracy
    final_baseline_performance = all_df['baseline_performance'].median() if 'baseline_performance' in all_df.columns and not all_df['baseline_performance'].isna().all() else baseline_performance

    if final_baseline_accuracy is None or np.isnan(final_baseline_accuracy):
        print("关键错误: 无法确定基线准确率。无法寻找最优节点数。")
        plot_results(all_df, args.output_dir, "all_no_baseline")
        return

    plot_results(all_df, args.output_dir, "all", final_baseline_accuracy, final_baseline_performance)
    
    optimal_nodes = find_optimal_edge_count_percentile(
        all_df, final_baseline_accuracy, 
        accuracy_percentile=args.accuracy_percentile, 
        f_percentile=args.f_percentile
    )
    
    print(f"\n{'='*80}")
    if optimal_nodes is not None:
        print(f"搜索完成！推荐的最佳边数: {int(optimal_nodes)}")
        optimal_row = all_df[all_df['n_edge'] == optimal_nodes]
        if not optimal_row.empty:
            opt_accuracy = optimal_row['accuracy'].iloc[0]
            opt_f = optimal_row['f'].iloc[0]
            opt_perf = optimal_row['performance'].iloc[0]
            acc_drop = final_baseline_accuracy - opt_accuracy
            print(f"该节点数下的准确率: {opt_accuracy:.4f} (较基线下降 {acc_drop:.4f})")
            print(f"该节点数下的性能: {opt_perf:.4f}")
            print(f"该节点数下的f值: {opt_f:.4f}")
    else:
        print("未能找到推荐的最佳节点数。")
    
    print(f"总耗时: {(time.time() - overall_start) / 60:.2f}分钟")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
