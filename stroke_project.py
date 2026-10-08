import numpy as np
import math

def compute_z(x, W, b):
    z = (W @ x + b).reshape(-1)
    return z


# -----------------------------------------------------------------
def compute_a(z):
    z_shifted = z - np.max(z)
    with np.errstate(under='ignore'):
        a = np.exp(z_shifted) / np.sum(np.exp(z_shifted))
    return a


# -----------------------------------------------------------------
def compute_L(a, y):
    ay = a[y]
    if ay <= 0:
        L = 1e10
    else:
        L = -float(np.log(ay))
    return L


# -----------------------------------------------------------------
def forward(x, y, W, b):
    z = compute_z(x, W, b)
    a = compute_a(z)
    L = compute_L(a, y)
    return z, a, L

def compute_dL_da(a, y):
    dL_da = np.zeros_like(a)
    if a[y] == 0:
        dL_da[y] = -1e10
    else:
        dL_da[y] = -1. / (a[y])
    return dL_da


# -----------------------------------------------------------------
def compute_da_dz(a):

    outer = np.outer(a, a)  # a_i * a_j everywhere, shape (3,3)
    diag = np.diag(a)
    da_dz = diag - outer
    #########################################
    return da_dz


# -----------------------------------------------------------------
def compute_dz_dW(x, c):
    dz_dW = np.tile(x[:, np.newaxis], (1, c)).T

    return dz_dW


# -----------------------------------------------------------------
def compute_dz_db(c):
    dz_db = np.ones(c)
    return dz_db

def backward(x, y, a):
    c = a.shape[0]
    dL_da = compute_dL_da(a, y)
    da_dz = compute_da_dz(a)
    dz_dW = compute_dz_dW(x, c)
    dz_db = compute_dz_db(c)
    return dL_da, da_dz, dz_dW, dz_db


# -----------------------------------------------------------------
def compute_dL_dz(dL_da, da_dz):
    dL_dz = dL_da @ da_dz
    #########################################
    return dL_dz


# -----------------------------------------------------------------
def compute_dL_dW(dL_dz, dz_dW):
    dL_dz = dL_dz.reshape(-1, 1)
    dL_dW = dz_dW * dL_dz
    return dL_dW


# -----------------------------------------------------------------
def compute_dL_db(dL_dz, dz_db):
    dL_db = dz_db * dL_dz
    return dL_db

# --------------------------
def update_W(W, dL_dW, alpha=0.001):
    W = W - alpha * dL_dW
    return W


# --------------------------
def update_b(b, dL_db, alpha=0.001):
    b = b - alpha * dL_db
    return b


# --------------------------
# train
def train(X, Y, alpha=0.01, n_epoch=100):
    # number of features
    p = X.shape[1]
    # number of classes
    c = max(Y) + 1

    # randomly initialize W and b
    W = np.random.rand(c, p)
    b = np.random.rand(c)

    for _ in range(n_epoch):
        # go through each training instance
        for x, y in zip(X, Y):
            z, a, L = forward(x, y, W, b)
            dL_da, da_dz, dz_dW, dz_db = backward(x, y, a)
            dL_dz = compute_dL_dz(dL_da, da_dz)
            dL_dW = compute_dL_dW(dL_dz, dz_dW)
            dL_db = compute_dL_db(dL_dz, dz_db)
            W = update_W(W, dL_dW, alpha)
            b = update_b(b, dL_db, alpha)

    return W, b


# --------------------------
def predict(Xtest, W, b):
    n = Xtest.shape[0]
    c = W.shape[0]
    Y = np.zeros(n, dtype=int)  # Initialize Y as integer array
    P = np.zeros((n, c))  # Initialize P with correct shape
    for i, x in enumerate(Xtest):
        #########################################
        ## INSERT YOUR CODE HERE
        z = compute_z(x, W, b)
        a = compute_a(z)
        Y[i] = np.argmax(a)
        P[i] = a
        #########################################
    return Y, P
