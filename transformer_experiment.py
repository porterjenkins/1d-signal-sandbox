from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sim import gen_1d_signal, plot_series
from utils import get_n_params, RunningAvgQueue
from datasets import TokenizedTimeSeriesDataset
from models.transformer import TransformerClassifier


# Build dataset
params = {
    "class_0": {
        "a": 2.0,
        "b": 0.5,
        "c": 1,
        "d": 0,
        "eps": 0.5,
    },
    "class_1": {
        "a": 1.8,
        "b": 0.5,
        "c": 1.0,
        "d": 0,
        "eps": 0.5,
    },
}
n = 1000
l = 100

X, y = gen_1d_signal(params, n, l)

x_c_0 = X[~y.astype(bool)]
x_c_1 = X[y.astype(bool)]
plot_series(x_c_0, x_c_1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# Training loop
n_epochs = 100
batch_size = 8
weight_decay = 0.01
lr = 3e-4

k = 20  # chunk/kernel size
d = 128  # hidden dim

model = TransformerClassifier.build(
    in_dim=k * 2, h_dim=d, attn_heads=8, encoder_blocks=1, max_seq_len=l // k
)

n_params = get_n_params(model)
print(f"Number of params: {n_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
bce_loss = nn.BCEWithLogitsLoss()

ts_trn = TokenizedTimeSeriesDataset(X_train, y_train, token_size=k, d_model=d)
trn_loader = DataLoader(ts_trn, shuffle=True, batch_size=16)

ts_test = TokenizedTimeSeriesDataset(X_test, y_test, token_size=k, d_model=d)
test_loader = DataLoader(ts_test, shuffle=True, batch_size=4)


loss_queue = RunningAvgQueue(maxsize=32)
acc_queue = RunningAvgQueue(maxsize=32)

loss_arr = []
acc_arr = []

test_acc_arr = []
for e in tqdm(range(n_epochs)):
    model.train()
    for input, target, pos in trn_loader:
        optimizer.zero_grad()
        y_hat = model(input, pos)

        loss = bce_loss(y_hat[:, 1], target)

        loss.backward()
        optimizer.step()

        loss_queue.add(float(loss.detach().data.numpy()))
        loss_arr.append(loss_queue.mean())

        acc = np.mean(
            (torch.argmax(F.softmax(y_hat, dim=-1), dim=-1) == target).data.numpy()
        )
        acc_queue.add(acc)

        acc_arr.append(acc_queue.mean())

    model.eval()
    test_y_hat_list = []
    test_target_list = []
    for input, target, pos in test_loader:
        y_hat = model(input, pos)

        test_y_hat_list.append(y_hat)
        test_target_list.append(target)

    test_y_hat_all = torch.cat(test_y_hat_list)
    test_target_all = torch.cat(test_target_list)

    test_acc = np.mean(
        (
            torch.argmax(F.softmax(test_y_hat_all, dim=-1), dim=-1) == test_target_all
        ).data.numpy()
    )
    test_acc_arr.append(test_acc)

plt.plot(np.arange(len(loss_arr)), loss_arr)
plt.xlabel("iteration")
plt.ylabel("train loss")
plt.title("Train Loss")
plt.show()

plt.plot(np.arange(len(acc_arr)), acc_arr)
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.title("Train Accuracy")
plt.axhline(acc_arr[-1], c="red", linestyle="--")
plt.show()


plt.plot(np.arange(len(test_acc_arr)), test_acc_arr)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.axhline(np.mean(test_acc_arr[-5:]), c="red", linestyle="--")
plt.title("Test Accuracy")
plt.show()


print("TEST ACCURACY: {:.4f}".format(np.mean(test_acc_arr[-5:])))
