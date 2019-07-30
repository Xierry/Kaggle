import numpy as np
from sklearn.utils import shuffle
from torch import nn , torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.optim.optimizer import Optimizer


class NeuralNet(nn.Module):
    def __init__(self, embedding, lstm_hidden_size = 120, gru_hidden_size = 60, p=0.2):
        super(NeuralNet, self).__init__()

        self.gru_hidden_size = gru_hidden_size
        self.embedding = nn.Embedding(*embedding.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding, dtype=torch.float32), requires_grad=False)

        self.embedding_dropout = nn.Dropout2d(p)

        self.lstm = nn.LSTM(embedding.shape[1], lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(gru_hidden_size * 2 * 3, 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(20, 1)

        self.loss_fn = nn.BCEWithLogitsLoss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        self.train()
    
    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def train_on_batch(self, x_batch, y_batch):
        
        self.optimizer.zero_grad()
        y_pred = self.forward(x_batch)
        loss = self.loss_fn(weight=y_batch[:, 1])(y_pred[:, 0], y_batch[:, 0])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        
        with torch.no_grad():
            self.eval()
            outs = self.forward(X).detach()
            preds = self.sigmoid(outs.cpu().numpy())[:, 0]
            self.train()
        return preds

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    def forward(self, x): 
        '''x: (B, T)'''
        
        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding) 
            # h_embedding: (B, T, D)

        h_lstm, _ = self.lstm(h_embedding)  # h_lstm: (B, T, lstm_hidden_size * 2)
        h_gru, hh_gru = self.gru(h_lstm)    # h_gru : (B, T, gru_hidden_size * 2)
        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)   # avg_pool: (B, gru_hidden_size*2)
        max_pool, _ = torch.max(h_gru, 1) # max_pool: (B, gru_hidden_size*2)
        
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
            # conc: (B, gru_hidden_size * 6)
        conc = self.relu(self.linear(conc))

        conc = self.dropout(conc)
        out  = self.out(conc)
        
        return out

class ParamScheduler():
    
    def __init__(self, optimizer, scale_fn, step_size):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
        self.optimizer = optimizer
        self.scale_fn = scale_fn
        self.step_size = step_size
        self.last_batch_iteration = 0
        
    def batch_step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scale_fn(self.last_batch_iteration / self.step_size)
        
        self.last_batch_iteration += 1

class EMA():
    # exponential-moving-average-decay
    def __init__(self, model, mu=0.9, level='batch', n=1, copy=False):
        # self.ema_model = copy.deepcopy(model) self.ema_model.eval()
        self.mu = mu
        self.level = level
        self.n = n # 用来计数几轮衰减
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data
    
    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone() # copy_ 还是detach?

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n
                
    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)

def collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        max_len = max(lens)

        padded_seqs = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            start = max_len - lens[i]
            padded_seqs[i, start:] = torch.LongTensor(seq)
        return padded_seqs

    index, seqs, targets = zip(*data)
    seqs = _pad_sequences(seqs)
    return index, seqs, torch.FloatTensor(targets)

class TextDataset(Dataset):

    def __init__(self, seqs, targets=None, maxlen=220):
        self.targets = targets if targets is not None else np.random.randint(2, size=(len(seqs),))
        self.seqs = seqs
        self.maxlen = maxlen

    def __len__(self):
        return len(self.seqs)

    def get_keys(self):
        lens = np.fromiter(
            (min(self.maxlen, len(seq)) for seq in self.seqs),
            dtype=np.int32)
        return lens

    def __getitem__(self, index):
        return index, self.seqs[index], self.targets[index]

class BucketSampler(Sampler):

    def __init__(self, data_source, sort_keys, bucket_size=None, batch_size=1048, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        self.weights = None

        if not shuffle_data:
            self.index = self.prepare_buckets()
        else:
            self.index = None

    def set_weights(self, weights):
        assert weights >= 0
        total = np.sum(weights)
        if total != 1:
            weights = weights / total
        self.weights = weights

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = self.prepare_buckets(indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)
        
    def prepare_buckets(self, indices=None):
        lens = - self.sort_keys
        assert self.bucket_size % self.batch_size == 0 or self.bucket_size == len(lens)

        if indices is None:
            if self.shuffle:
                indices = shuffle(np.arange(len(lens), dtype=np.int32))
                lens = lens[indices]
            else:
                indices = np.arange(len(lens), dtype=np.int32)

        #  bucket iterator
        def divide_chunks(l, n):
            if n == len(l):
                yield np.arange(len(l), dtype=np.int32), l
            else:
                # looping till length l
                for i in range(0, len(l), n):
                    data = l[i:i + n]
                    yield np.arange(i, i + len(data), dtype=np.int32), data
    
        new_indices = []
        extra_batch = None
        for chunk_index, chunk in divide_chunks(lens, self.bucket_size):
            # sort indices in bucket by descending order of length
            indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
            batches = []
            for _, batch in divide_chunks(indices_sorted, self.batch_size):
                if len(batch) == self.batch_size:
                    batches.append(batch.tolist())
                else:
                    assert extra_batch is None
                    assert batch is not None
                    extra_batch = batch
    
            # shuffling batches within buckets
            if self.shuffle:
                batches = shuffle(batches)
            for batch in batches:
                new_indices.extend(batch)
    
        if extra_batch is not None:
            new_indices.extend(extra_batch)
        return indices[new_indices]
        
    