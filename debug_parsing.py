
import re
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def normalize_token(token):
    return token.strip().lower()

def _parse_and_suffix_tokens(original_sentence_text: str, marked_sentence: str, increase_markers: tuple = ("<<", ">>"), decrease_markers: tuple = ("[[", "]]")) -> tuple[list[str], list[str]]:
    # 1. 预计算全局token计数，用于判断是否需要加后缀
    original_tokens = word_tokenize(original_sentence_text)
    global_counts = defaultdict(int)
    for token in original_tokens:
        global_counts[normalize_token(token)] += 1

    # 2. 使用NLTK对带标记的句子进行分词
    marked_tokens = word_tokenize(marked_sentence)
    print(f"DEBUG: marked_tokens = {marked_tokens}")

    # 3. 初始化状态机和结果列表
    increase_suffixed, decrease_suffixed = [], []
    running_counts = defaultdict(int)
    
    in_increase = False
    in_decrease = False
    last_token = "" # 用于检测连续标记

    # 4. 【已修正】遍历分词后的token列表，使用更清晰的状态机逻辑
    for token in marked_tokens:
        # a. 更新状态机
        
        # 检查是否是开始标记 (支持单token '<<' 或 双token '<' '<')
        is_inc_start = (token == increase_markers[0]) or (token == increase_markers[0][0] and last_token == increase_markers[0][0])
        is_dec_start = (token == decrease_markers[0]) or (token == decrease_markers[0][0] and last_token == decrease_markers[0][0])
        
        # 检查是否是结束标记 (支持单token '>>' 或 双token '>' '>')
        is_inc_end = (token == increase_markers[1]) or (token == increase_markers[1][0] and last_token == increase_markers[1][0])
        is_dec_end = (token == decrease_markers[1]) or (token == decrease_markers[1][0] and last_token == decrease_markers[1][0])

        if is_inc_start:
            in_increase = True
            last_token = token
            continue 
        elif is_dec_start:
            in_decrease = True
            last_token = token
            continue
        elif is_inc_end:
            in_increase = False
            last_token = token
            continue
        elif is_dec_end:
            in_decrease = False
            last_token = token
            continue

        # b. 如果在标记内部，则处理当前token
        # 确保当前token不是标记本身
        # 这里的检查主要是为了防止 NLTK 分词产生的单个 < 或 [ 字符被误认为是内容
        is_marker_char = token in ['<', '>', '[', ']']
        if not is_marker_char and (in_increase or in_decrease):
            norm_token = normalize_token(token)
            running_counts[norm_token] += 1
            
            # c. 根据全局计数决定是否添加后缀
            if global_counts.get(norm_token, 0) > 1:
                suffixed_token = f"{token.strip()}_{running_counts[norm_token]}"
            else:
                suffixed_token = token.strip()
            
            # d. 将带后缀的token添加到对应的结果列表
            if in_increase:
                increase_suffixed.append(suffixed_token)
            elif in_decrease:
                decrease_suffixed.append(suffixed_token)
        
        last_token = token

    return increase_suffixed, decrease_suffixed

# Case from user
sid = "ioi_0088"
sentence_text = "After the meeting, Emily and Lewis reviewed the agenda. Emily handed a folder to"
marked_line = "ioi_0088: After the meeting, <<Lewis>> and [[Emily]] reviewed the agenda. Emily handed a folder to"

inc, dec = _parse_and_suffix_tokens(sentence_text, marked_line)
print(f"Increase: {inc}")
print(f"Decrease: {dec}")

