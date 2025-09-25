import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from q3_1 import gradient_descent_ridge
from q1_1 import data_matrix_bias, predict, rmse

X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.flatten()
y_test = pd.read_csv("y_test.csv").values.flatten()

X_train = data_matrix_bias(X_train)
X_test = data_matrix_bias(X_test)

schedules = ["constant", "exp_decay", "cosine"]
eta0 = 0.001
k = 0.001
T= 100
lamb = 1.0

for schedule in schedules:
    w, L = gradient_descent_ridge(X_train, y_train, lamb=lamb, eta0=eta0, T=T, schedule=schedule, k_decay=k)

    plt.plot(range(T), L, label= schedule)
    
    y_hat = predict(X_test, w)
    print(f"{schedule} RMSE:", rmse(y_test, y_hat))
    

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Iterations")
plt.legend()
plt.savefig("report/assets/training_loss_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
