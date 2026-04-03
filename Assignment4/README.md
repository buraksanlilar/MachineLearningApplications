# Assignment 4

This assignment evaluates a manually implemented multiple linear regression model using repeated validation and k-fold cross-validation.

## What was done

- The dataset path in `lab4.py` was updated to read `Football_players.csv` from the project root (same style as Assignment 3).
- The duplicate dataset file under `Assignment4/` was removed to avoid keeping two copies.
- Features were built manually as:
  - Intercept term (ones)
  - Age
  - Height
  - Mental
  - Skill
- Target variable is salary.
- Validation error is computed with an 80/20 split after shuffling.
- 8-fold cross-validation is implemented manually:
  - Test set is the current fold.
  - Training set is the concatenation of all other folds.
- Mean Squared Error (MSE) is implemented manually.
- Regression coefficients are computed with the normal equation:
  - w = (X^T X)^(-1) X^T y

## Run

From the project root:

```bash
python3 Assignment4/lab4.py
```

The script prints validation MSE and 8-fold CV MSE for 10 shuffled runs.
