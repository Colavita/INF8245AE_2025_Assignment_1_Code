from q2_1 import *
import pandas as pd
import numpy as np

def print_result(metric, best_lamb, scores):
    print("Metric: ", metric)
    print("Best Lambda: ", best_lamb)
    print("Best score: ", scores[best_lamb])
    print("Scores: ", scores)
    print("-" * 100)

X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values
y_test = pd.read_csv("y_test.csv").values

lambda_list = [0.01, 0.1, 1, 10 ,100]
metrics = ["MAE", "MaxError", "RMSE"]
n_folds = 5

results = {}
for metric in metrics:
    best_lambda, scores = cross_validate_ridge(X_train, y_train, lambda_list, n_folds, metric)
    print_result(metric, best_lambda, scores)
    results[metric] = (best_lambda, scores)