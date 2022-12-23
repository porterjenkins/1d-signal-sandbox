from torch.utils.data import Dataset
import torch
import math


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        super(TimeSeriesDataset, self).__init__()
        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __getitem__(self, idx):
        x_i = torch.tensor(self.X[idx, :]).float()
        y_i = torch.tensor(self.y[idx])

        x_i = torch.permute(x_i, (1, 0))

        return x_i, y_i

    def __len__(self):
        return self.n


class TokenizedTimeSeriesDataset(Dataset):

    def __init__(self, X, y, token_size, d_model):
        super(TokenizedTimeSeriesDataset, self).__init__()
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.l = X.shape[1]
        self.max_seq_len = self.l // token_size
        self.token_size = token_size
        self.d_model = d_model

    def __getitem__(self, idx):
        x_i = torch.tensor(self.X[idx, :]).float()
        y_i = torch.tensor(self.y[idx])

        x_i = torch.permute(x_i, (1, 0))

        x_i = torch.stack(torch.split(x_i, self.token_size, dim=1)).flatten(1)
        #pos = torch.arange(0, self.max_seq_len, dtype=torch.long)
        pos = self.get_pos_encoding(self.max_seq_len, self.d_model)

        return x_i, y_i, pos

    @staticmethod
    def get_pos_encoding(max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        return pe

    def __len__(self):
        return self.n