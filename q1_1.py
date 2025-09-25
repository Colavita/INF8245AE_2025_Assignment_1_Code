import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    n = X.shape[0]
    X_bias = np.hstack((np.ones((n, 1)), X))
    return X_bias

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS solution"""
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    return np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    lambda_matrix = np.diag(lambda_vec)
    return np.linalg.inv(X.T @ X + lambda_matrix) @ X.T @ y

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    return X @ w

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    errors = y - y_hat
    rmse = np.sqrt(np.mean(errors**2))
    return rmse