import json
import glob
import os
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
topn_values = []
edges_list = []
results_list = []
total_edge = 32923
with open('../results/ioi/result_ioi_topn.json', 'r') as f:
      results = json.load(f)
for result in results:
      topn_values.append(result['topn'])
      edges_list.append(result['edges'])
      results_list.append(result['results'])

# 定义原始数据
acdc_original_edge_sparsity = [0.9708, 0.973, 0.976, 0.9785, 0.982, 0.983, 0.9845, 0.986, 0.9885]
acdc_original_result = [0.25, 0.3, 0.27, 0.42, 0.47, 0.52, 0.57, 0.63, 0.8]
eap_original_edge_sparsity = [0.93, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]
eap_original_result = [1.38, 1.6, 1.95, 2.7, 3.2, 3.35, 3.5, 3.68, 3.55]

# 计算 edge sparsity

test_edge_sparsity = [ 1 - edge / total_edge for edge in edges_list]

# 绘制图片
fig, ax = plt.subplots()
ax.scatter(test_edge_sparsity, results_list, color='blue', label='EAP-IG-inputs')
ax.scatter(eap_original_edge_sparsity, eap_original_result, color='red', label='EAP_original')
ax.grid(True)
ax.set_title('Scatter Plot of Two EAP Groups')
ax.set_xlabel('Edge Sparsity Axis')
ax.set_ylabel('kl_divergence Axis')
ax.legend()

# 保存图片到上级目录的 results/ioi 文件夹中
output_dir = os.path.join(current_dir, '../results/ioi')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'ioi_plot_eap-ig.png'))
plt.show()