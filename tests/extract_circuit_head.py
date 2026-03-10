def extract_head(data, sentence): 
      layers = {i: [] for i in range(12)}

# 遍历所有节点
      for node, info in data['nodes'].items():
            if node.startswith('a') and info['in_graph']:
                  # 提取层号和head号
                  layer, head = node.split('.')
                  layer_num = int(layer[1:])
                  head_num = int(head[1:])
                  # 将head添加到对应的层
                  layers[layer_num].append(head_num)

      # 打开文件以追加模式写入
      with open('results/ioi/circuits/head.txt', 'a') as file:
      # 对每一层的head进行排序并写入文件
            file.write(sentence + '\n')
            for layer_num in range(12):
                  heads = sorted(layers[layer_num])
                  # 将head号转换为字符串格式，如11.4
                  heads_str = [f"{layer_num}.{head}" for head in heads]
                  file.write(' '.join(heads_str) + '\n')