import csv
import argparse
import sys
import difflib

def similarity_ratio(s1, s2):
    """
    使用difflib计算两个字符串的相似度比例
    """
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def deduplicate_csv(input_path: str, output_path: str, column_index: int = 0, threshold: float = 0.7):
    """
    读取CSV文件，对所有句子对进行全局去重。
    检查每一对与之前所有保留的对的相似度，如果超过阈值则移除。
    """
    print(f"正在处理输入文件: {input_path}")
    
    unique_rows = []
    unique_clean_texts = []  # 存储已保留的clean文本，用于比较
    
    total_rows = 0
    total_pairs = 0
    kept_pairs = 0
    removed_pairs = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # 首先读取并保存表头
            header = next(reader)
            unique_rows.append(header)
            
            # 读取所有数据行
            all_rows = list(reader)
            total_rows = len(all_rows)
            
            i = 0
            while i < len(all_rows):
                # 确保有足够的行可以组成一对
                if i + 1 < len(all_rows):
                    total_pairs += 1
                    
                    # 获取这一对的两行数据
                    row1 = all_rows[i]
                    row2 = all_rows[i+1]
                    
                    # 获取clean列文本用于比较
                    if len(row1) > column_index:
                        clean_text = row1[column_index]
                        
                        # 检查是否与已保存的任何clean文本高度相似
                        is_duplicate = False
                        max_similarity = 0.0
                        similar_text = ""
                        
                        for existing_text in unique_clean_texts:
                            sim = similarity_ratio(clean_text, existing_text)
                            if sim > max_similarity:
                                max_similarity = sim
                                similar_text = existing_text
                            
                            if sim >= threshold:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            # 保存这一对数据
                            unique_rows.append(row1)
                            unique_rows.append(row2)
                            unique_clean_texts.append(clean_text)
                            kept_pairs += 1
                            print(f"保留对 {kept_pairs}: {clean_text}")
                        else:
                            removed_pairs += 1
                            print(f"移除重复对: {clean_text}")
                            print(f"  -> 与已保留文本相似 (相似度: {max_similarity:.3f}): {similar_text}")
                    else:
                        # 如果列索引无效，默认保留
                        unique_rows.append(row1)
                        unique_rows.append(row2)
                        kept_pairs += 1
                        print(f"警告: 行 {i+1} 缺少列 {column_index}，默认保留")
                    
                    # 移动到下一对
                    i += 2
                else:
                    # 如果最后剩一行无法组成对，也保留它
                    unique_rows.append(all_rows[i])
                    i += 1
    
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        sys.exit(1)

    # 将去重后的数据写入输出文件
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(unique_rows)
        
    print("\n处理完成！")
    print(f"总共处理了 {total_rows} 行数据 ({total_pairs} 对)。")
    print(f"去重后保留 {kept_pairs} 对数据 ({len(unique_rows)-1} 行，不含表头)。")
    print(f"移除了 {removed_pairs} 对重复数据。")
    print(f"重复率: {removed_pairs/total_pairs*100:.1f}%")
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用difflib对EAP-IG任务的CSV文件进行全局去重。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="需要去重的源CSV文件路径。"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="保存去重后数据的目标CSV文件路径。"
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="用于判断重复的列的索引 (0代表第一列，即'clean'列)。"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="相似度阈值，高于此阈值的句子对会被视为重复（范围：0-1）。"
    )
    
    args = parser.parse_args()
    
    deduplicate_csv(args.input, args.output, args.column, args.threshold)