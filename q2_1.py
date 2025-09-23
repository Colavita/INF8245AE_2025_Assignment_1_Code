import numpy as np
from q1_1 import rmse, ridge_regression_optimize, data_matrix_bias, predict


# Part (a)
def cv_splitter(X, y, k):
    """
    Splits data into k folds for cross-validation.
    Returns a list of tuples: (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    """
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

    folds = []

    current_index = 0
    for fold_size in fold_sizes:
        start = current_index
        end = current_index + fold_size

        val_index = np.arange(start, end)
        train_index = np.concatenate([np.arange(0, start), np.arange(end, n)])

        X_val_fold = X_shuffled[val_index]
        y_val_fold = y_shuffled[val_index]
        X_train_fold = X_shuffled[train_index]
        y_train_fold = y_shuffled[train_index]

        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        current_index = end

    return folds

# Part (b)
def MAE(y, y_hat):
    return np.mean(np.abs(y-y_hat))

def MaxError(y, y_hat):
    return np.max(np.abs(y-y_hat))

# Part (c)
def cross_validate_ridge(X, y, lambda_list, k, metric):
    """
    Performs k-fold CV over lambda_list using the given metric.
    metric: one of "MAE", "MaxError", "RMSE"
    Returns the lambda with best average score and a dictionary of mean scores.
    """
    folds = cv_splitter(X, y, k)

    scores = {}
    for lamb in lambda_list:
        fold_errors = []
        for fold in folds:
            X_train, y_train, X_val, y_val = fold

            X_train = data_matrix_bias(X_train)
            X_val = data_matrix_bias(X_val)

            w = ridge_regression_optimize(X_train, y_train, lamb=lamb)
            y_hat = predict(X_val, w)

            if metric == "MAE":
                err = MAE(y_val, y_hat)
            elif metric == "MaxError":
                err = MaxError(y_val, y_hat)
            elif metric == "RMSE":
                err = rmse(y_val, y_hat)

            fold_errors.append(err)

        avg_error = np.mean(fold_errors)
        scores[lamb] = avg_error

    best_lambda = min(scores.keys(), key=lambda x: scores[x])

    return best_lambda, scores