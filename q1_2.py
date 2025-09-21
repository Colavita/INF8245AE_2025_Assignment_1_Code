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

X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values
y_test = pd.read_csv("y_test.csv").values

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

def plot_results(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color="red", linestyle="--")
    plt.show()

plot_results(y_test, y_hat_ols, f"OLS (RMSE={rmse_ols:.3f})")
plot_results(y_test, y_hat_ridge, f"Ridge (RMSE={rmse_ridge:.3f})")
plot_results(y_test, y_hat_weighted_ridge, f"Weighted Ridge (RMSE={rmse_weighted_ridge:.3f})")
