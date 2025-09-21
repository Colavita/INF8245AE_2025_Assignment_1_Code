import numpy as np


# Part (a)
def data_matrix_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias column of ones as the first column of X."""
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias

# Part (b)
def linear_regression_optimize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    #TODO: Sometimes, XTX is not invertible. 
    """Closed-form OLS solution"""
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    w = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y 
    return w

# Part (e)
def weighted_ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
    """Weighted Ridge regression solution."""
    lambda_matrix = np.diag(lambda_vec)
    w = np.linalg.inv(X.T @ X + lambda_matrix) @ X.T @ y
    return w


# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    y_hat = X @ w
    return y_hat

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    errors = y - y_hat
    rmse = np.sqrt(np.mean(errors**2))
    return rmse