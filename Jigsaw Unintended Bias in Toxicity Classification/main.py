import os, re, copy,random,string,time,warnings,joblib,numpy as np,pandas as pd
from collections import Counter
from contextlib import contextmanager 
from functools import partial
from itertools import chain
from multiprocessing import Pool

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch, torch.nn as nn
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.optim.optimizer import Optimizer

from utils import *
from models import *
warnings.filterwarnings('ignore')

# 用到的常变量
EMBEDDING_FASTTEXT = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
TRAIN_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_DATA = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
SAMPLE_SUBMISSION = '../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'

embed_size = 300
max_features = 100000
max_len = 220
batch_size = 512
train_epochs = 6
n_splits = 5
updates_per_epoch = 10

seed_torch()
device = torch.device('cuda: 0')
ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer('english')
misspell_dict= {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                "didn't": "did not", "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is",
                "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                "she'd": "she would", "she'll": "she will", "she's": "she is",
                "shouldn't": "should not", "that's": "that is", "there's": "there is",
                "they'd": "they would", "they'll": "they will", "they're": "they are",
                "they've": "they have", "we'd": "we would", "we're": "we are",
                "weren't": "were not", "we've": "we have", "what'll": "what will",
                "what're": "what are", "what's": "what is", "what've": "what have",
                "where's": "where is", "who'd": "who would", "who'll": "who will",
                "who're": "who are", "who's": "who is", "who've": "who have",
                "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are", "you've": "you have",
                "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}
puncts=[',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
        '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
        '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
        '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
        '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
        '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
        'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
        '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

# 数据处理部分
train_x, train_y, train_y_identity, test_x, nan_pred, train_nan_mask, test_nan_mask, y_binary, y_identity_binary, vocab = load_and_prec()
    # train_x: (1804874,); train_y: (1804874, 2), train_y_identity: (1804874, 9),test_x: (97320,)

embedding_matrix = load_embedding(EMBEDDING_FASTTEXT, vocab['token2id'])

train_dataset = TextDataset(train_x, targets=train_y, maxlen=max_len)
test_dataset  = TextDataset(test_x , maxlen=max_len)

train_sampler = BucketSampler(train_dataset, train_dataset.get_keys(),
                                bucket_size=batch_size * 20, batch_size=batch_size)
test_sampler  = BucketSampler(test_dataset, test_dataset.get_keys(),
                                batch_size=batch_size, shuffle_data=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                            sampler=train_sampler, num_workers=0, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)

# 训练

train_preds = np.zeros((len(train_x)))
test_preds  = np.zeros((len(test_x)))

models={} # 存储模型参数
model = NeuralNet(embedding_matrix).to(device)

ema_model = copy.deepcopy(model) # 深拷贝一份模型
ema_model.eval()                 # 设置成验证模式
ema = EMA(model, n=int(len(train_loader.dataset) / (10 * batch_size))) # 指数移动平均

scale_fn = combine_scale_functions(
    [partial(scale_cos, 1e-4, 5e-3), partial(scale_cos, 5e-3, 1e-3)], [0.2, 0.8])

scheduler = ParamScheduler(model.optimizer, scale_fn, train_epochs * len(train_loader))

all_test_preds = [] # 存储每个epoch模型对测试集的预测
all_losses = []     # 存储每个epoch的batch_loss列表 二维列表

for epoch in range(train_epochs):
    start_time = time.time()
    model.train()
    epoch_losses=[]

    for _, x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)        # print('x_batch: ', x_batch.size())

        scheduler.batch_step()

        loss_batch = model.train_on_batch(x_batch, y_batch)
        epoch_losses.append(loss_batch)

        ema.on_batch_end(model) # 更新参数
        
    all_losses.append(epoch_losses)
    test_preds = eval_model(model, test_loader)
    all_test_preds.append(test_preds)

    ema.on_epoch_end(model)

    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t time={:.2f}s'.format(
            epoch + 1, train_epochs, elapsed_time))

ema.set_weights(ema_model) # 训练参数复制到备份模型中

checkpoint_weights = np.array([2 ** epoch for epoch in range(train_epochs)])
checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()

test_y = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
test_y[test_nan_mask] = nan_pred

models['model'] = model.state_dict()
models['ema_model'] = ema_model.state_dict()

