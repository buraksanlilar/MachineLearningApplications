import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import linear_model


def MSE(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / len(y_true)

def multlin_coef(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def ridge_coef(X, y, alpha):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    return np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y


def main():
    assignment_dir = Path(__file__).resolve().parent
    root_dir = assignment_dir.parent
    csv_path = root_dir / 'Football_players.csv'

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                         usecols=(4, 5, 6, 7, 8), dtype=float, encoding='latin-1')

    age    = data[:, 0]
    height = data[:, 1]
    mental = data[:, 2]
    skill  = data[:, 3]
    salary = data[:, 4]

    ones = np.ones(len(data))
    X = np.column_stack((ones, age, height, mental, skill))

    coefficients = multlin_coef(X, salary)
    y_pred = X @ coefficients
    mse = MSE(salary, y_pred)
    print("Test error with multiple linear regression:\nMSE:", mse)

    ridge_coefficients = ridge_coef(X, salary, alpha=5.0)
    y_pred_ridge = X @ ridge_coefficients
    mse_ridge = MSE(salary, y_pred_ridge)
    print("\nTest error with ridge regression:\nMSE:", mse_ridge)

    train_size = len(data) - 20
    X_train = X[:train_size]
    X_test  = X[train_size:]
    y_train = salary[:train_size]
    y_test  = salary[train_size:]

    coefficients_train = multlin_coef(X_train, y_train)
    y_pred_train = X_test @ coefficients_train
    mse_train = MSE(y_test, y_pred_train)
    print("\nTest error of multiple linear regression w/ 80-20 split:\nMSE:", mse_train)

    coefficients_ridge_train = ridge_coef(X_train, y_train, alpha=5.0)
    y_pred_ridge_train = X_test @ coefficients_ridge_train
    mse_ridge_train = MSE(y_test, y_pred_ridge_train)
    print("\nTest error of ridge regression w/ 80-20 split:\nMSE:", mse_ridge_train)

    # --- scikit-learn: sadece age ve salary ---
    X_sklearn       = age.reshape(-1, 1)         
    y_sklearn       = salary
    X_train_sklearn = X_sklearn[:train_size]
    X_test_sklearn  = X_sklearn[train_size:]
    y_train_sklearn = y_sklearn[:train_size]
    y_test_sklearn  = y_sklearn[train_size:]

    model_lr = linear_model.LinearRegression()
    model_lr.fit(X_train_sklearn, y_train_sklearn)
    model_lr_pred = model_lr.predict(X_test_sklearn)

    model_rg = linear_model.Ridge(alpha=50)
    model_rg.fit(X_train_sklearn, y_train_sklearn)
    model_rg_pred = model_rg.predict(X_test_sklearn)

    model_lasso = linear_model.Lasso(alpha=50)
    model_lasso.fit(X_train_sklearn, y_train_sklearn)
    model_lasso_pred = model_lasso.predict(X_test_sklearn)

    # --- Grafikler ---
    x_line = np.linspace(age.min(), age.max(), 200).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(age[:train_size], y_train_sklearn, color='blue', alpha=0.6, label='Train Data (80%)')
    ax.scatter(age[train_size:], y_test_sklearn,  color='red',  alpha=0.8, label='Test Data (20%)')
    ax.plot(x_line, model_lr.predict(x_line), color='black', linewidth=2, label='Linear (No Reg)')
    ax.plot(x_line, model_rg.predict(x_line), color='cyan',  linewidth=2, linestyle='--', label='Ridge (L2)')
    ax.set_title('Ridge vs Linear Regression')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary')
    ax.legend()

    ax = axes[1]
    ax.scatter(age[:train_size], y_train_sklearn, color='blue',   alpha=0.6, label='Train Data (80%)')
    ax.scatter(age[train_size:], y_test_sklearn,  color='red',    alpha=0.8, label='Test Data (20%)')
    ax.plot(x_line, model_lr.predict(x_line),    color='black',  linewidth=2, label='Linear (No Reg)')
    ax.plot(x_line, model_lasso.predict(x_line), color='orange', linewidth=2, linestyle='--', label='Lasso (L1)')
    ax.set_title('Lasso vs Linear Regression')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary')
    ax.legend()

    plt.tight_layout()
    plot_path = assignment_dir / 'lab5_plots.png'
    plt.savefig(plot_path, dpi=150)
    print("\nPlot saved:", plot_path)
    plt.show()


if __name__ == "__main__":
    main()