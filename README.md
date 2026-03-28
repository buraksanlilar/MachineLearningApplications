# CE475 Machine Learning Applications

This repository contains my hands-on work for the CE475 Machine Learning course.
It includes assignment implementations, lab exercises, and supporting notes as the semester progresses.

## Repository Structure

- `Assignment0/`: Introductory NumPy exercises.
- `Assignment1/`: Simple Linear Regression implementation and report-oriented documentation.
- `Assignment2/`: Simple Linear Regression with train/test split and feature comparison.
- `Assignment3/`: Multiple Linear Regression and MSE-based evaluation.

## Completed So Far

- Basic array and matrix operations with NumPy
- Random number generation and summary statistics
- Manual implementation of Simple Linear Regression coefficients (`b0`, `b1`)
- Visualization of regression results using Matplotlib
- Train/test split practice (`80/20`) with model comparison on different features
- Manual implementation of Multiple Linear Regression coefficients using normal equation
- Mean Squared Error (MSE) evaluation on test and full data
- Effect analysis of adding a random extra feature column

## Assignment Summary

- `Assignment0`: NumPy warm-up tasks for array manipulation and statistics.
- `Assignment1`: `Age -> Salary` simple linear regression (manual coefficient calculation + plot).
- `Assignment2`: Two simple regression models (`Age -> Salary`, `Height -> Salary`) with train/test split visualization.
- `Assignment3`: Multiple regression with `Age`, `Height`, `Mental`, `Skill` to predict `Salary`, then MSE comparison after adding a random column.

## Tech Stack

- Python 3
- NumPy
- Matplotlib

## How to Run

Run each assignment from its own folder.

Example:

```bash
cd Assignment3
python3 lab3.py
```

## Goal of This Repository

To keep a clean, organized, and reproducible record of my CE475 Machine Learning lab progress from foundational topics to more advanced applications.
