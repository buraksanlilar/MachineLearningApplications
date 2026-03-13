# Assignment 2 - Simple Linear Regression with Train/Test Split

This assignment extends the previous linear regression task by separating the dataset into training and test sets and comparing two different input features against salary.

## What This Lab Does

- Reads `Football_players.csv`
- Extracts the `Age`, `Height`, and `Salary` columns as NumPy arrays
- Splits the dataset into:
  - training data: all rows except the last 20
  - test data: the last 20 rows
- Computes simple linear regression coefficients manually for:
  - `Age` vs `Salary`
  - `Height` vs `Salary`
- Displays both regression results in a single figure using subplots

## Implemented Functions

- `simlin_coef(x, y)`
  - Calculates the regression coefficients `b0` and `b1` manually using the simple linear regression formula

- `simlin_plot(ax, x_train, y_train, x_test, y_test, b0, b1, xlabel)`
  - Draws the regression line
  - Plots training data in blue
  - Plots test data in red
  - Adds labels, title, and legend

## Visualization

The output figure contains two subplots:

1. `Age` vs `Salary`
2. `Height` vs `Salary`

Each subplot includes:

- black regression line
- blue training points
- red test points

## Files

- `lab2.py`: main Python script for the assignment
- `Football_players.csv`: dataset used in the analysis

## How to Run

```bash
cd Assignment2
python3 lab2.py
```

## Purpose

The goal of this lab is to practice:

- working with NumPy arrays
- splitting data into training and test sets
- manually computing linear regression coefficients
- comparing different input features visually with Matplotlib
