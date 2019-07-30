import os, re, copy,random,string,time,warnings,joblib,numpy as np,pandas as pd,_pickle
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

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')
def build_vocab(texts, max_features):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {
        'token2id': {'<PAD>': 0, '<UNK>': max_features + 1},
        'id2token': {}
    }
    vocab['token2id'].update(
                        {token: _id + 1 for _id, (token, count) in 
                            enumerate(counter.most_common(max_features))}
    )
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    return vocab
def tokenize(texts, vocab):
    
    def text2ids(text, token2id):
        return [
            token2id.get(token, len(token2id) - 1)
            for token in text.split()[:max_len]]
    
    return [
        text2ids(text, vocab['token2id'])
        for text in texts]
def load_and_prec():
    train = pd.read_csv(TRAIN_DATA, index_col='id'); train =train.sample(100000).reset_index(drop=True)
    test = pd.read_csv(TEST_DATA, index_col='id')
    
    # lower
    train['comment_text'] = train['comment_text'].str.lower()
    test['comment_text']  = test['comment_text'].str.lower()

    # clean misspellings
    train['comment_text'] = train['comment_text'].apply(replace_typical_misspell)
    test['comment_text'] = test['comment_text'].apply(replace_typical_misspell)

    # clean the text
    train['comment_text'] = train['comment_text'].apply(clean_text)
    test['comment_text'] = test['comment_text'].apply(clean_text)

    # clean numbers
    train['comment_text'] = train['comment_text'].apply(clean_numbers)
    test['comment_text'] = test['comment_text'].apply(clean_numbers)
    
    # strip
    train['comment_text'] = train['comment_text'].str.strip()
    test['comment_text'] = test['comment_text'].str.strip()
        
    # replace blank with nan
    train['comment_text'].replace('', np.nan, inplace=True)
    test['comment_text'].replace('', np.nan, inplace=True)

    # nan prediction
    nan_pred = train['target'][train['comment_text'].isna()].mean()
    
    # fill up the missing values
    train_x = list(train['comment_text'].fillna('_##_'))
    test_x = list(test['comment_text'].fillna('_##_'))
    
    # get the target values
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    weights = np.ones((len(train),))
    weights += train[identity_columns].fillna(0).values.sum(axis=1) * 3
    weights += train['target'].values * 8
    weights /= weights.max()
    train_y = np.vstack([train['target'].values, weights]).T
    
    train_y_identity = train[identity_columns].values


    train_nan_mask = [True if text == '_##_' else False for text in train_x]
    test_nan_mask  = [True if text == '_##_' else False for text in test_x ]

    y_binary = (train_y[:, 0] >= 0.5).astype(int) # (1804874, 2)
    y_identity_binary = (train_y_identity >= 0.5).astype(int) # (1804874, 9)

    vocab = build_vocab(train_x + test_x, max_features)

    train_x = tokenize(train_x, vocab)
    test_x  = tokenize(test_x,  vocab)
    return train_x, train_y, train_y_identity, test_x, nan_pred, train_nan_mask, test_nan_mask, y_binary, y_identity_binary, vocab
def load_embedding(embedding_path, word_index):

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.strip().split(' ')) for o in open(embedding_path))
    
    # word_index = tokenizer.word_index
    nb_words = min(max_features + 2, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))

    for key, i in word_index.items():
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue

    return embedding_matrix
def _get_misspell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re
def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell(misspell_dict)

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)
def clean_text(x):
    x = str(x)
    for punct in puncts + list(string.punctuation):
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x
def clean_numbers(x):
    return re.sub(r'\d+', ' ', x)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def eval_model(model, data_loader):
    preds = np.zeros(len(data_loader.dataset))    
    with torch.no_grad():
        for index, x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)            
            y_pred = model.predict(x_batch)
            preds[list(index)] = y_pred
    return preds
def scale_cos(start, end, x):
    return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2
def combine_scale_functions(scale_fns, phases=None):
    if phases is None:
        phases = [1. / len(scale_fns)] * len(scale_fns)
    phases = [phase / sum(phases) for phase in phases]
    phases = torch.tensor([0] + phases)
    phases = torch.cumsum(phases, 0)
    
    def _inner(x):
        idx = (x >= phases).nonzero().max()
        actual_x = (x - phases[idx]) / (phases[idx + 1] - phases[idx])
        return scale_fns[idx](actual_x)
        
    return _inner

def save_pkl(obj, file = './obj.pkl'):
    with open(file, 'wb') as f:        
        _pickle.dump(obj, f, -1)

def load_pkl(file=None):
    try:
        with open(file, 'rb') as f:  # 读取
            return _pickle.load(f) 
    except: print('load error')