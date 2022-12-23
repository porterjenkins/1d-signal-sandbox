import numpy as np
import matplotlib.pyplot as plt


def get_sin(x, a, b, c, d):
    y = a*np.sin(b*(x+c)) + d
    return y


def plot_series(X_a, X_b):
    fig = plt.figure()
    l_a = X_a.shape[1]
    l_b = X_b.shape[1]

    plt.plot(X_a[:, : 0].mean(0), c='blue', linestyle='--', alpha=0.5)
    for i in range(X_a.shape[0]):
        plt.plot(np.arange(l_a), X_a[i, :, 0], c='blue', alpha=0.01)

    plt.plot(X_b[:, : 0].mean(0), c='orange', linestyle='--', alpha=.5)
    for i in range(X_b.shape[0]):
        plt.plot(np.arange(l_b), X_b[i, :, 0], c='orange', alpha=0.01)

    plt.show()



def gen_1d_signal(params: dict, n: int = 1000, l: int = 100):
    x = np.arange(l)
    X = np.zeros((n, l))
    y = np.zeros(n)

    for i in range(n):
        alpha = np.random.random()
        if alpha < 0.5:
            # class 0
            a = params['class_0']['a']
            b = params['class_0']['b']
            c = params['class_0']['c']
            d = params['class_0']['d']

            eps = params['class_0']['eps']
            y_i = 0.0
        else:
            # class 1
            a = params['class_1']['a']
            b = params['class_1']['b']
            c = params['class_1']['c']
            d = params['class_1']['d']

            eps = params['class_1']['eps']
            y_i = 1.0

        signal = get_sin(x, a, b, c, d) + np.random.normal(0, eps, size=l)
        X[i, :] = signal
        y[i] = y_i

    temp_diff = np.diff(X, n=1, axis=-1)
    X_signal = np.zeros((n, l, 2))
    # two channels: 1) signal, 2) derivative
    X_signal[:, :, 0] = X
    X_signal[:, 1:, 1] = temp_diff

    return X_signal, y

if __name__ == "__main__":

    X, y = gen_1d_signal()
    x_c_0 = X[~y.astype(bool)]
    x_c_1 = X[y.astype(bool)]

    plot_series(x_c_0, x_c_1)



