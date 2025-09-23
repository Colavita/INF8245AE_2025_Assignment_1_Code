from q2_1 import *
import pandas as pd
import numpy as np

X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values
y_test = pd.read_csv("y_test.csv").values

lambda_list = [0.01, 0.1, 1, 10 ,100]
metrics = ["MAE", "MaxError", "RMSE"]

results = {}
for metric in metrics:
    best_lambda, scores = cross_validate_ridge(X_train, y_train, lambda_list, 5, metric)
    results[metric] = (best_lambda, scores)

print(results)

#    Metric  Best λ  λ=0.01  λ=0.1   λ=1   λ=10  λ=100
#       MAE      10    7.446  7.343  7.128  7.103  7.805
#  MaxError      10   27.441 27.316 26.941 26.797 27.503
#      RMSE       1    9.701  9.639  9.537  9.547 10.086