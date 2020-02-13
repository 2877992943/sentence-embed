from laserembeddings import Laser

laser = Laser()

# if all sentences are in the same language:

# embeddings = laser.embed_sentences(
#     ['let your neural network be polyglot',
#      'use multilingual embeddings!'],
#     lang='en')  # lang is only used for tokenization
#
# print ('')


embeddings = laser.embed_sentences(
    ['今 天 天 气 晴 朗',
     '今 天 天 气 很 不 错',
     '股 票 怎 么 跌 成 这 样'],
    lang='en')  # lang is only used for tokenization


# embeddings = laser.embed_sentences(
#     ['今天天气晴朗',
#      '今天天气很不错',
#      '股票怎么跌成这样'],
#     lang='zh')  #使用jieba分词# lang is only used for tokenization
#
# print ('')

# embeddings is a N*1024 (N = number of sentences) NumPy array

import pandas as pdd
pdd.to_pickle(embeddings,'emb.pkl')