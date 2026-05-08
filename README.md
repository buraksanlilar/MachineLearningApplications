# CE475 Machine Learning Applications

Hands-on lab work for the CE475 Machine Learning course. Each assignment builds on the previous one, progressing from basic regression to ensemble methods.

## Repository Structure

| Folder | Topic |
|---|---|
| `Assignment0/` | NumPy warm-up exercises |
| `Assignment1/` | Simple Linear Regression (manual) |
| `Assignment2/` | Simple Linear Regression with train/test split |
| `Assignment3/` | Multiple Linear Regression + MSE analysis |
| `Assignment4/` | Repeated validation and k-fold cross-validation |
| `Assignment5/` | Regularization: Ridge & Lasso vs manual regression |
| `Assignment6/` | Polynomial Regression (degree 2 and 3) |
| `Assignment7/` | Random Forest Regression with hyperparameter sweep |

## Assignment Summary

- **Assignment0**: Array manipulation and statistics with NumPy.
- **Assignment1**: `Age -> Salary` simple linear regression — manual `b0`, `b1` calculation and plot.
- **Assignment2**: Two regression models (`Age -> Salary`, `Height -> Salary`) with 80/20 train/test split and visualization.
- **Assignment3**: Multiple regression with `Age`, `Height`, `Mental`, `Skill` predicting `Salary`; MSE comparison before/after adding a random feature.
- **Assignment4**: Manual validation with shuffling and 8-fold cross-validation on the same multiple regression setup.
- **Assignment5**: Manual multiple/ridge regression alongside scikit-learn `LinearRegression`, `Ridge`, `Lasso`; saved comparison plots.
- **Assignment6**: Manual polynomial regression (degree 2 and 3) for `Age` and `Skill` features; 3D surface plots.
- **Assignment7**: `RandomForestRegressor` hyperparameter sweep over `n_estimators` (10–100) and `max_depth` (3–12); OOB MSE evaluation; 3D scatter plot of results.

## Tech Stack

- Python 3
- NumPy
- Matplotlib
- scikit-learn

## How to Run

All assignments can be run from the repository root:

```bash
python3 Assignment1/lab1.py
python3 Assignment2/lab2.py
python3 Assignment3/lab3.py
python3 Assignment4/lab4.py
python3 Assignment5/lab5.py
python3 Assignment6/lab6.py
python3 Assignment7/lab7.py
```

Or from each assignment's folder:

```bash
cd Assignment7
python3 lab7.py
```

## Dataset

All assignments from Assignment1 onward use the shared `Football_players.csv` at the repository root. Scripts reference it via a relative path from their location — no need to copy or move the file.

## Goal

Clean, reproducible record of CE475 ML lab progress — from basic regression fundamentals to ensemble learning.
