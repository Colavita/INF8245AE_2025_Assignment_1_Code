import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from q1_1 import (
    data_matrix_bias,
    linear_regression_optimize,
    ridge_regression_optimize,
    weighted_ridge_regression_optimize,
    predict,
    rmse
)

def generate_plots(y_test, y_hat_ols, y_hat_ridge, y_hat_weighted_ridge, rmse_ols, rmse_ridge, rmse_weighted_ridge):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    y_test_flat = y_test.flatten()
    y_hat_ols_flat = y_hat_ols.flatten()
    y_hat_ridge_flat = y_hat_ridge.flatten()
    y_hat_weighted_ridge_flat = y_hat_weighted_ridge.flatten()

    axes[0].scatter(y_test_flat, y_hat_ols_flat, alpha=0.6, color='blue')
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"OLS (RMSE={rmse_ols:.3f})")
    axes[0].plot([y_test_flat.min(), y_test_flat.max()],
                [y_test_flat.min(), y_test_flat.max()],
                color="red", linestyle="--")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test_flat, y_hat_ridge_flat, alpha=0.6, color='green')
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Ridge (RMSE={rmse_ridge:.3f})")
    axes[1].plot([y_test_flat.min(), y_test_flat.max()],
                [y_test_flat.min(), y_test_flat.max()],
                color="red", linestyle="--")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(y_test_flat, y_hat_weighted_ridge_flat, alpha=0.6, color='orange')
    axes[2].set_xlabel("Actual")
    axes[2].set_ylabel("Predicted")
    axes[2].set_title(f"Weighted Ridge (RMSE={rmse_weighted_ridge:.3f})")
    axes[2].plot([y_test_flat.min(), y_test_flat.max()],
                [y_test_flat.min(), y_test_flat.max()],
                color="red", linestyle="--")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as 'regression_comparison.png'")

if __name__ == "__main__":

    X_train = pd.read_csv("X_train.csv").to_numpy()
    X_test = pd.read_csv("X_test.csv").to_numpy()
    y_train = pd.read_csv("y_train.csv").to_numpy()
    y_test = pd.read_csv("y_test.csv").to_numpy()

    X_train = data_matrix_bias(X_train)
    X_test = data_matrix_bias(X_test)

    w_ols = linear_regression_optimize(X_train, y_train)
    w_ridge = ridge_regression_optimize(X_train, y_train, lamb=1.0)
    lambda_vec =  np.array([0.01, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
    w_weighted_ridge = weighted_ridge_regression_optimize(X_train, y_train, lambda_vec)

    y_hat_ols = predict(X_test, w_ols)
    y_hat_ridge = predict(X_test, w_ridge)
    y_hat_weighted_ridge = predict(X_test, w_weighted_ridge)

    rmse_ols = rmse(y_test, y_hat_ols)
    rmse_ridge = rmse(y_test, y_hat_ridge)
    rmse_weighted_ridge = rmse(y_test, y_hat_weighted_ridge)

    print("OLS RMSE:", rmse_ols)
    print("RIDGE RMSE:", rmse_ridge)
    print("WEIGHTED RIDGE RMSE:", rmse_weighted_ridge)

    generate_plots(y_test, y_hat_ols, y_hat_ridge, y_hat_weighted_ridge, rmse_ols, rmse_ridge, rmse_weighted_ridge)

