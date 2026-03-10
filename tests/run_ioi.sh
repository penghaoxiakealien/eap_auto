#!/bin/bash

# 定义 topn 参数列表
#topn_values=(200 300)
topn_values=(250 225 275 175 150)
export CUDA_VISIBLE_DEVICES=7
# 遍历 topn 参数并运行 Python 脚本
for topn in "${topn_values[@]}"; do
    echo "Running experiment with topn=$topn"
    python ioi.py $topn
done

# 收集所有结果并绘制图片
echo "Collecting results and plotting..."
python plot_ioi_kl.py