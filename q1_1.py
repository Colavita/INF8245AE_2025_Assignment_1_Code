import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    n = X.shape[0]
    bias = np.ones((n, 1), dtype=X.dtype)
    return np.hstack((bias, X))

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return np.ravel(w)

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    w = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y
    return np.ravel(w)

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    lambda_vec_d = lambda_vec.shape[0]
    X_d = X.shape[1]

    if lambda_vec_d < X_d:
        print(f"Warning: lambda_vec shorter than feature dimension ({lambda_vec_d} < {X_d}). Padding with zeros.")
        lambda_vec = np.concatenate([lambda_vec, np.zeros(X_d - lambda_vec_d)])
    elif lambda_vec_d > X_d:
        print(f"Warning: lambda_vec longer than feature dimension ({lambda_vec_d} > {X_d}). Truncating extra values.")
        lambda_vec = lambda_vec[:X_d]
    
    lambda_matrix = np.diag(lambda_vec)
    w = np.linalg.inv(X.T @ X + lambda_matrix) @ X.T @ y
    return np.ravel(w)

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    X = np.asarray(X)
    w = np.asarray(w)
    y_hat = X @ w
    return np.ravel(y_hat)

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    errors = y - y_hat
    return float(np.sqrt(np.mean(errors**2)))