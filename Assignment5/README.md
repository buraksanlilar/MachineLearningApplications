# Assignment 5 - Regularized Regression with Scikit-learn

In this assignment, we compare the manual multiple linear regression approach used in previous labs
with ready-to-use models from scikit-learn.
On the same dataset, Linear Regression, Ridge (L2), and Lasso (L1) models are trained
and analyzed visually.

## Objective

- Work on a player salary prediction problem using football player data
- Compare manual solutions (normal equation) and library-based solutions in one script
- Understand the effect of regularization (penalization)
- Interpret model behavior with MSE values and plots

## Dataset

This assignment now uses the shared dataset in the project root
instead of a duplicated CSV file inside this folder:

- File used: ../Football_players.csv
- Columns used:
  - Age
  - Height
  - Mental
  - Skill
  - Salary (target)

In code, the path is resolved with Path:

- Assignment folder: the folder where this script is located
- Root folder: the parent of the assignment folder
- CSV path: root/Football_players.csv

This keeps dataset management centralized and avoids repeated CSV copies across assignments.

## What We Implemented

1. Loaded the data and selected the required numeric columns.
2. Computed manual multiple linear regression coefficients using the normal equation.
3. Computed manual Ridge coefficients with an alpha parameter.
4. Printed MSE values for both full-data and 80/20 split settings.
5. Trained the following scikit-learn models using Age -> Salary:
   - LinearRegression
   - Ridge
   - Lasso
6. Plotted fitted model lines together with train/test points.
7. Saved the final figure under the Assignment5 folder.

## Modeling Approach

### 1) Manual Multiple Linear Regression

Coefficient estimation is done with the normal equation:

w = (X^T X)^(-1) X^T y

Here, a bias column (ones) is added to matrix X.

### 2) Manual Ridge Regression

The regularized normal equation is used:

w = (X^T X + alpha I)^(-1) X^T y

To avoid penalizing the bias term, the first diagonal element of I is set to 0.

### 3) Scikit-learn Models

The Age feature is used alone, and with an 80/20 split these models are fitted:

- LinearRegression()
- Ridge(alpha=50)
- Lasso(alpha=50)

Then test predictions are generated and plotted.

## Plot Output

The generated plot file is saved in this folder:

- lab5_plots.png

The save path is explicitly configured in code to target the Assignment5 folder.

## How To Run

From the project root:

```bash
python3 Assignment5/lab5.py
```

or from inside Assignment5:

```bash
cd Assignment5
python3 lab5.py
```

In both cases:

- the data source is Football_players.csv in the repository root
- the plot output is created as Assignment5/lab5_plots.png

## Expected Outputs

When the program runs, it prints these metrics in the terminal:

- Manual multiple linear regression MSE
- Manual ridge regression MSE
- Test MSE values for both models under the 80/20 split

It also generates a two-panel visualization:

- Left panel: Linear vs Ridge
- Right panel: Linear vs Lasso

## Summary

This assignment demonstrates regularization both at the equation level
and through practical scikit-learn implementations.
With centralized data access and assignment-scoped plot output,
the workflow is cleaner and avoids file duplication.
