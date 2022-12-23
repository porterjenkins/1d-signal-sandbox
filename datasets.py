from torch.utils.data import Dataset
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F

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