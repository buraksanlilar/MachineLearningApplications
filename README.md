# CE475 Machine Learning Applications

This repository contains my hands-on work for the CE475 Machine Learning course.
It includes assignment implementations, lab exercises, and supporting notes as the semester progresses.

## Repository Structure

- `Assignment0/`: Introductory NumPy exercises.
- `Assignment1/`: Simple Linear Regression implementation and report-oriented documentation.
- `Assignment2/`: Simple Linear Regression with train/test split and feature comparison.
- `Assignment3/`: Multiple Linear Regression and MSE-based evaluation.
- `Assignment4/`: Repeated validation and k-fold cross-validation for Multiple Linear Regression.
- `Assignment5/`: Manual vs scikit-learn Linear/Ridge/Lasso comparison with visualization.
- `Assignment6/`: Manual polynomial regression with degree 2 and degree 3 models.

## Completed So Far

- Basic array and matrix operations with NumPy
- Random number generation and summary statistics
- Manual implementation of Simple Linear Regression coefficients (`b0`, `b1`)
- Visualization of regression results using Matplotlib
- Train/test split practice (`80/20`) with model comparison on different features
- Manual implementation of Multiple Linear Regression coefficients using normal equation
- Mean Squared Error (MSE) evaluation on test and full data
- Effect analysis of adding a random extra feature column
- Repeated validation with shuffling and 8-fold cross-validation on the same regression setup
- Ridge and Lasso regularization comparison against linear regression
- scikit-learn based regression workflow integrated with manual implementations
- Centralized dataset usage from repository root and assignment-specific plot output
- Manual polynomial regression with degree 2 and degree 3 feature expansion

## Assignment Summary

- `Assignment0`: NumPy warm-up tasks for array manipulation and statistics.
- `Assignment1`: `Age -> Salary` simple linear regression (manual coefficient calculation + plot).
- `Assignment2`: Two simple regression models (`Age -> Salary`, `Height -> Salary`) with train/test split visualization.
- `Assignment3`: Multiple regression with `Age`, `Height`, `Mental`, `Skill` to predict `Salary`, then MSE comparison after adding a random column.
- `Assignment4`: Manual validation and 8-fold cross-validation for the same multiple regression model, with repeated shuffling and MSE comparison.
- `Assignment5`: Manual multiple regression and ridge alongside scikit-learn `LinearRegression`, `Ridge`, and `Lasso`, including saved comparison plots.
- `Assignment6`: Manual polynomial regression for `Age` and `Skill` features to predict `Salary`, with degree 2 and degree 3 comparisons.

## Tech Stack

- Python 3
- NumPy
- Matplotlib
- scikit-learn

## How to Run

Run each assignment from its own folder.

Example:

```bash
cd Assignment3
python3 lab3.py
```

For Assignment4:

```bash
cd Assignment4
python3 lab4.py
```

For Assignment5:

```bash
python3 Assignment5/lab5.py
```

Notes for Assignment5:

- Uses the shared dataset at `Football_players.csv` in the repository root.
- Saves plot output to `Assignment5/lab5_plots.png`.

Notes for Assignment6:

- Uses the shared dataset at `Football_players.csv` in the repository root.
- Reads the dataset from the assignment script using a path relative to the project root.
- Includes `Assignment6/README.md` for the assignment-specific explanation and run instructions.

## Goal of This Repository

To keep a clean, organized, and reproducible record of my CE475 Machine Learning lab progress from foundational topics to more advanced applications.
