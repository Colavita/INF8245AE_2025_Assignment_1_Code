import numpy as np


# Part (a)
def ridge_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb: float) -> np.ndarray:
    """
    Computes the gradient of Ridge regression loss.
    ∇L(w) = -2/n X^T (y - X w) + 2 λ w
    """
    n = X.shape[0]
    grad = (-2/n) * (X.T @ (y - X @ w)) + 2 * lamb * w
    return grad


# Part (b)
def learning_rate_exp_decay(eta0: float, t: int, k_decay: float) -> float:
    return eta0 * np.exp(-k_decay * t)


# Part (c)
def learning_rate_cosine_annealing(eta0: float, t: int, T: int) -> float:
    return eta0 * 0.5 * (1 + np.cos((np.pi * t) / T))


# Part (d)
def gradient_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, lamb:float, eta: float) -> np.ndarray:
    grad = ridge_gradient(X, y, w, lamb=lamb)
    return w - (eta * grad)
    

# Part (e)
def gradient_descent_ridge(X, y, lamb=1.0, eta0=0.1, T=500, schedule="constant", k_decay=0.01):
    n = len(X)
    d = X.shape[1]
    w = np.zeros(d)
    L = np.zeros(T)
    for t in range(T):
        if schedule == "constant":
            eta = eta0
        elif schedule == "exp_decay":
            eta = learning_rate_exp_decay(eta0, t, k_decay)
        elif schedule == "cosine":
            eta = learning_rate_cosine_annealing(eta0, t, T)
        
        w = gradient_step(X, y, w, lamb, eta)
        y_hat = X @ w
        errors = y - y_hat
        L[t] = ((1/n) * np.sum(errors**2)) + (lamb * np.sum(w**2))

    return w, L
