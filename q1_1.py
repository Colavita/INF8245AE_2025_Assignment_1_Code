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
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# Part (c)
def ridge_regression_optimize(X: np.ndarray, y: np.ndarray, lamb: float) -> np.ndarray:
    """Closed-form Ridge regression."""
    return np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y

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
    return np.linalg.inv(X.T @ X + lambda_matrix) @ X.T @ y

# Part (f)
def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute predictions: y_hat = X w"""
    return X @ w

# Part (f)
def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error"""
    errors = y - y_hat
    return float(np.sqrt(np.mean(errors**2)))

def main():
    # === Test 1: Single sample, multiple features ===
    X1 = np.array([[2.0, 3.0]])
    y1 = np.array([1.0])
    X1_bias = data_matrix_bias(X1)
    w1 = linear_regression_optimize(X1_bias, y1)
    print("Test 1 (single sample):", w1)

    # === Test 2: Single feature, multiple samples ===
    X2 = np.array([[1.0], [2.0], [3.0]])
    y2 = np.array([2.0, 4.0, 6.0])
    X2_bias = data_matrix_bias(X2)
    w2 = linear_regression_optimize(X2_bias, y2)
    print("Test 2 (single feature):", w2)

    # === Test 3: Collinear features (non-invertible X^T X) ===
    X3 = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)  # second column = 2 * first column
    y3 = np.array([1.0, 2.0, 3.0])
    X3_bias = data_matrix_bias(X3)
    try:
        w3 = linear_regression_optimize(X3_bias, y3)
        print("Test 3 (collinear, linear regression):", w3)
    except np.linalg.LinAlgError as e:
        print("Test 3 failed (linear regression):", e)

    # Ridge regression should fix the singularity
    w3_ridge = ridge_regression_optimize(X3_bias, y3, lamb=1e-3)
    print("Test 3 (collinear, ridge):", w3_ridge)

    # === Test 4: Weighted Ridge with some zero weights ===
    lambda_vec = np.array([0.0, 1.0])  
    w4 = weighted_ridge_regression_optimize(X2_bias, y2, lambda_vec)
    print("Test 4 (weighted ridge, fixed):", w4)

    # Mismatch on purpose to check error handling
    try:
        bad_lambda_vec = np.array([0.0, 1.0, 10.0])  
        weighted_ridge_regression_optimize(X2_bias, y2, bad_lambda_vec)
    except ValueError as e:
        print("Caught mismatch error as expected:", e)

    # === Test 5: Prediction & RMSE ===
    y_hat = predict(X2_bias, w2)
    print("Predictions:", y_hat)
    print("RMSE:", rmse(y2, y_hat))

    # === Test 6: Edge case empty array (should handle gracefully or error clearly) ===
    try:
        X_empty = np.empty((0, 2))
        y_empty = np.empty((0,))
        X_empty_bias = data_matrix_bias(X_empty)
        print("Test 6 (empty):", X_empty_bias.shape)
    except Exception as e:
        print("Test 6 failed (empty input):", e)


if __name__ == "__main__":
    main()