import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
csv_path = root_dir / 'Football_players.csv'

data = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                     usecols=(4, 5, 6, 7, 8), dtype=float, encoding='latin-1')

X = data[:, :4]   # Age, Height, Mental, Skill
y = data[:, 4]    # Salary

train_size = len(data) - 20
X_train = X[:train_size]
X_test  = X[train_size:]
y_train = y[:train_size]
y_test  = y[train_size:]

results_n   = []
results_d   = []
results_mse = []

for n in range(10, 101):
    for d in range(3, 13):
        try:
            warnings.filterwarnings('error')
            reg = RandomForestRegressor(n_estimators=n, max_depth=d, oob_score=True)
            reg.fit(X_train, y_train)
            oob_predictions = reg.oob_prediction_
            mse = np.mean((y_train - oob_predictions) ** 2)
            results_n.append(n)
            results_d.append(d)
            results_mse.append(mse)
            print(f"n={n}, d={d}, MSE={mse:.2f}")
        except Exception:
            pass
        finally:
            warnings.resetwarnings()

results_n   = np.array(results_n)
results_d   = np.array(results_d)
results_mse = np.array(results_mse)

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(results_d, results_n, results_mse, c='red', s=10)

min_idx = np.argmin(results_mse)
ax.scatter(results_d[min_idx], results_n[min_idx], results_mse[min_idx],
           c='blue', s=40, zorder=5)

ax.set_xlabel('Depth')
ax.set_ylabel('Number of trees')
ax.set_zlabel('MSE')
ax.set_title('Random Forest: n, d vs MSE')
plt.tight_layout()
plt.show()

# Depth analysis
print()
for d in range(3, 13):
    mask = results_d == d
    avg_mse = np.mean(results_mse[mask])
    print(f"Average MSE for d = {d}: {avg_mse:.2f}")
