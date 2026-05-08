# Assignment7 - Random Forest Regression

This assignment explores Random Forest Regression using scikit-learn to predict football player salaries. It sweeps over a range of hyperparameters and visualizes their effect on model performance.

## How It Works

1. Loads `Football_players.csv` from the repository root.
2. Uses four features: `Age`, `Height`, `Mental`, `Skill` to predict `Salary`.
3. Splits the data: last 20 samples as test set, rest as training.
4. Trains a `RandomForestRegressor` for every combination of:
   - `n_estimators` (number of trees): 10 to 100
   - `max_depth` (tree depth): 3 to 12
5. Evaluates each model using **OOB (Out-of-Bag) MSE** on the training set.
6. Produces a **3D scatter plot** of depth vs. number of trees vs. MSE (best point highlighted in blue).
7. Prints average MSE per depth level for a summary analysis.

## Run

From the repository root:

```bash
python3 Assignment7/lab7.py
```

Or from this folder:

```bash
python3 lab7.py
```

## Notes

- Dataset is read from the repository root (`../Football_players.csv`).
- OOB score requires `n_estimators` to be large enough; combinations that fail are silently skipped.
- The 3D plot shows the full hyperparameter search space; the blue dot marks the minimum MSE configuration.
