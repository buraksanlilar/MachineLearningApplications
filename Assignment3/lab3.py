import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path(__file__).resolve().parents[1] / 'Football_players.csv'
    data = np.genfromtxt(csv_path, delimiter=',',skip_header=1,usecols=(4,5,6,7,8),dtype=None,encoding='latin-1')
    
    age = np.array(data[:,0]) #age
    height = np.array(data[:,1]) #height
    mental = np.array(data[:,2]) #mental
    skill = np.array(data[:,3]) #skill
    ones = np.ones(len(data)) #ones
    salary = np.array(data[:,4]) #salary

    X = np.column_stack((ones,age,height,mental,skill))

    train_size = len(data) - 20
    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = salary[:train_size]
    y_test = salary[train_size:]

    print("---- Original Data ----\n")
    coefficients = multlin_coef(X_train, y_train)
    y_pred = X_test @ coefficients
    mse = MSE(y_test, y_pred)
    print("Test error w/ 80/20 split:\nMSE:", mse)
    coefficients_full = multlin_coef(X, salary)
    y_pred_full = X @ coefficients_full
    mse_full = MSE(salary, y_pred_full)
    print("\nTraining error\nMSE:", mse_full)

    print("\n---- DATA W/ RANDOM COLN ----")
    extra_colmn = np.random.randint(low=-1000, high=1001, size=len(data)) #random column with values between -1000 and 1000
    X_with_extra = np.column_stack((X, extra_colmn))
    X_train_extra = X_with_extra[:train_size]
    X_test_extra = X_with_extra[train_size:]
    coefficients_extra = multlin_coef(X_train_extra, y_train)
    y_pred_extra = X_test_extra @ coefficients_extra
    mse_extra = MSE(y_test, y_pred_extra)
    print("Test error w/ 80/20 split:\nMSE:", mse_extra)

    coefficients_full_extra = multlin_coef(X_with_extra, salary)
    y_pred_full_extra = X_with_extra @ coefficients_full_extra
    mse_full_extra = MSE(salary, y_pred_full_extra)
    print("\nTraining error\nMSE:", mse_full_extra)

def MSE(y_true, y_pred):
    return np.sum(np.square(y_true - y_pred)) / len(y_true)

def multlin_coef(X,y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

if __name__ == "__main__":
    main()

