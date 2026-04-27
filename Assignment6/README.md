# Assignment6 - Polynomial Regression (Degree 2 and 3)

This assignment implements polynomial regression manually using NumPy to model player salary based on selected football player features.

## What Was Changed

- The script now reads the dataset from the repository root (`../Football_players.csv`) instead of keeping a duplicate CSV file inside `Assignment6`.
- The local CSV file in this folder was removed to keep dataset usage centralized and avoid inconsistencies.

## How It Works

1. The script loads the root dataset and selects these columns:
   - `Age`
   - `Skill`
   - `Salary`
2. It builds polynomial feature matrices manually for:
   - Degree 2
   - Degree 3
3. It computes regression coefficients using the normal equation:
   - $w = (X^T X)^{-1} X^T y$
4. It predicts salary for both models and computes Mean Squared Error (MSE).
5. It visualizes both fitted surfaces in separate 3D plots.

## Run

From the repository root:

```bash
python3 Assignment6/lab6.py
```

Or from this folder:

```bash
python3 lab6.py
```

## Notes

- Keep `Football_players.csv` only in the repository root.
- If you add new assignments, follow the same centralized dataset pattern.
