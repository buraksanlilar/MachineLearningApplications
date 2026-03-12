# Assignment 1 - Simple Linear Regression

This folder contains the solution for the **Simple Linear Regression** task described in the LAB 1 document.

## Task Summary (From PDF)

1. Read the provided `Football_players.csv` file.
2. Extract the `Age` and `Salary` columns into separate NumPy arrays.
3. Implement two functions in the same file:
   - `simlin_coef(x, y)`: Computes regression coefficients and returns `(b0, b1)`.
   - `simlin_plot(x, y, b0, b1)`: Plots real data as a scatter plot and the regression line as a line plot on the same figure.
4. Compute the coefficients manually using the given formulas, not with ready-made libraries (for example, sklearn).

## Formulas Used

- `y_hat = b1 * x + b0`
- `b1 = sum((xi - x_mean) * (yi - y_mean)) / sum((xi - x_mean)^2)`
- `b0 = y_mean - b1 * x_mean`

## Files

- `main.py`: Handles CSV reading, coefficient calculation, and plotting.
- `Football_players.csv`: Sample dataset.
- `LAB 1 - Instructions.pdf`: Original assignment instructions (not required to upload to the repository).

## Run

```bash
python3 main.py
```

After running:

- blue points: real data (`Age`, `Salary`)
- red line: simple linear regression result
