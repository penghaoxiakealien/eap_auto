#!/bin/bash

# 脚本出错时立即退出
set -e
set -o pipefail

python test_single_hypothesis.py \
--layer 7 \
--head 3 \
--num_sentences 10 \
--dataset_split validation \
--hypothesis "** Sender Head (7, 3) functions primarily to suppress attention to the subject or agent in a sentence, thereby promoting attention to the recipient or indirect object. This suppression ensures that downstream heads, such as Head 9.6 and Head 9.9, accurately focus on the recipient as the indirect object in the IOI task. By inhibiting attention to the subject, Sender Head (7, 3) facilitates the correct identification of the recipient, allowing downstream heads to allocate their attention effectively and maintain the intended sentence structure."