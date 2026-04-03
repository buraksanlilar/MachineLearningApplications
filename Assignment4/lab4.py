import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path(__file__).resolve().parents[1] / 'Football_players.csv'
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(4,5,6,7,8), dtype=None, encoding='latin-1')

    age    = np.array(data[:, 0])
    height = np.array(data[:, 1])
    mental = np.array(data[:, 2])
    skill  = np.array(data[:, 3])
    salary = np.array(data[:, 4])
    ones   = np.ones(len(data))

    X = np.column_stack((ones, age, height, mental, skill))
    y = salary

    # Birleşik matris: X ve y'yi aynı anda shuffle edebilmek için
    Xy = np.column_stack((X, y))

    print(f"{'Validation MSE':<20} {'8-fold CV MSE'}")
    print(f"{'-'*14:<20} {'-'*13}")

    for _ in range(10):
        # Shuffle (X ve y aynı sırada)
        np.random.shuffle(Xy)
        X_shuf = Xy[:, :-1]
        y_shuf = Xy[:, -1]

        # Validation MSE (80/20 split)
        train_size = len(Xy) - 20
        X_train, X_test = X_shuf[:train_size], X_shuf[train_size:]
        y_train, y_test = y_shuf[:train_size], y_shuf[train_size:]

        coef   = multlin_coef(X_train, y_train)
        y_pred = X_test @ coef
        val_mse = MSE(y_test, y_pred)

        # 8-fold CV MSE
        cv_mse = cross_validation(X_shuf, y_shuf, k=8)

        print(f"{val_mse:<20.2f} {cv_mse:.2f}")

def MSE(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / len(y_true)

def multlin_coef(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def cross_validation(X, y, k):
    if k <= 1:
        raise ValueError("k must be greater than 1")
    if k > len(X):
        raise ValueError("k cannot be greater than number of samples")

    indices = np.arange(len(X))
    folds   = np.array_split(indices, k)
    mse_list = []

    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_test,  y_test  = X[test_idx],  y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]

        coef   = multlin_coef(X_train, y_train)
        y_pred = X_test @ coef
        mse_list.append(MSE(y_test, y_pred))

    return np.mean(mse_list)

if __name__ == "__main__":
    main()
