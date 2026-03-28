# Assignment 3 - Multiple Linear Regression and Error Comparison

This assignment builds a multiple linear regression model to predict player salary from multiple features in `Football_players.csv`.

## What This Lab Does

- Reads the shared dataset from the repository root
- Uses these input features:
  - `Age`
  - `Height`
  - `Mental`
  - `Skill`
- Uses `Salary` as the target variable
- Builds a design matrix with a bias (ones) column
- Splits data into:
  - training set: all rows except the last 20
  - test set: last 20 rows

## Implemented Functions

- `multlin_coef(X, y)`
  - Computes multiple linear regression coefficients using the normal equation:
  - `(X^T X)^-1 X^T y`

- `MSE(y_true, y_pred)`
  - Calculates Mean Squared Error for model evaluation

## Experiments Included

1. **Original feature set** (`Age`, `Height`, `Mental`, `Skill`)
   - Test MSE on 80/20 split
   - Training MSE on full dataset

2. **Feature set with an extra random column**
   - Adds one random feature with values between `-1000` and `1000`
   - Repeats test and training MSE calculation
   - Compares the effect of adding an unrelated feature

## Files

- `lab3.py`: assignment script
- `README.md`: assignment explanation

## How to Run

```bash
cd Assignment3
python3 lab3.py
```

## Purpose

The goal is to practice:

- manual multiple linear regression implementation
- matrix-based coefficient computation
- train/test evaluation with MSE
- understanding how irrelevant features can affect model behavior
