import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
csv_path = root_dir / 'Football_players.csv'

data = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                     usecols=(4, 5, 6, 7, 8), dtype=float, encoding='latin-1')

x1 = data[:, 0]  # Age
x2 = data[:, 3]  # Skill
y  = data[:, 4]  # Salary


def polynomial_matrix(x1, x2, k):
    columns = []
    for i in range(k + 1):
        for j in range(k + 1):
            if i + j <= k:
                col = (x1 ** i) * (x2 ** j)
                columns.append(col)
    X = np.column_stack(columns)
    return X


def coefficients(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def compute_mse(y, y_hat):
    n = len(y)
    return np.sum((y - y_hat) ** 2) / n


# ── Degree 2 ────────────────────────────
X2 = polynomial_matrix(x1, x2, k=2)
w2 = coefficients(X2, y)
y_hat2 = X2 @ w2
mse2 = compute_mse(y, y_hat2)
print(f"MSE with degree 2: {mse2}")

# ── Degree 3 ────────────────────────────
X3 = polynomial_matrix(x1, x2, k=3)
w3 = coefficients(X3, y)
y_hat3 = X3 @ w3
mse3 = compute_mse(y, y_hat3)
print(f"MSE with degree 3: {mse3}")


# ── Plotting ──────────────────────────────────────────────────────────────────
# Figure 1 – Degree 2
fig1 = plt.figure(1, figsize=(9, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x1, x2, y, color='red', s=40, zorder=5)
ax1.plot_trisurf(x1, x2, y_hat2, alpha=0.5, color='gray')
ax1.set_xlabel('Age')
ax1.set_ylabel('Skill')
ax1.set_zlabel('Salary')
ax1.set_title('Degree-2 Polynomial Regression')

# Figure 2 – Degree 3
fig2 = plt.figure(2, figsize=(9, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x1, x2, y, color='red', s=40, zorder=5)
ax2.plot_trisurf(x1, x2, y_hat3, alpha=0.5, color='gray')
ax2.set_xlabel('Age')
ax2.set_ylabel('Skill')
ax2.set_zlabel('Salary')
ax2.set_title('Degree-3 Polynomial Regression')

plt.show()