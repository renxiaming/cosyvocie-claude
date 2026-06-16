# cosyvoice/utils/sentence_splitter.py
import re

# 可在此处按需调整标点符号，包括中英文句号、问号、感叹号、逗号、分号等
PUNCTUATIONS = r'[。，！？；、]' 

def split_text_by_punctuation(text: str) -> list:
    """
    按标点符号分割文本，返回句子列表。
    会保留文本中的标点符号，并自动处理空白符。
    """
    # 使用正则表达式的捕获组 (capturing group) 来保留分隔符
    # 同时修剪掉可能出现的空白符
    sentences = re.split(f'({PUNCTUATIONS})', text)
    # 将句子和标点重新组合，并去除空白符
    final_sentences = []
    for i in range(0, len(sentences), 2):
        if i+1 < len(sentences):
            # 组合句子 + 标点
            combined = sentences[i] + sentences[i+1]
        else:
            # 处理文本末尾可能没有标点的情况
            combined = sentences[i]
        if combined.strip():
            final_sentences.append(combined.strip())
    return final_sentences