import pandas as pd
import numpy as np
import seaborn as sns
import string
import nltk
from nltk.corpus import words,stopwords


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_words= ' '.join(filtered_words)
    return filtered_words

def is_spelled_correctly(word):
    english_words = set(words.words())
    return word in english_words

def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')

    return text

def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text



if __name__ == '__main__':
    df = pd.read_csv('../data_set/AI_Human.csv')
    # 去除换行等存储结构带来的字符影响
    df['text'] = df['text'].apply(remove_tags)
    # 去除符号干扰
    df['text'] = df['text'].apply(remove_punc)
    # 过滤掉对文本分析无实质贡献的常见词
    df['text'] = df['text'].apply(remove_stopwords)
    df.to_csv('../data_set/processed_AI_Human.csv',
              index=False,  # 不保存行索引
              encoding='utf-8',  # 保持原始编码
              sep=',',  # 使用原始分隔符
              header=True)  # 保留列标题
