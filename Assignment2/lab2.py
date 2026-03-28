import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path(__file__).resolve().parents[1] / 'Football_players.csv'
    data = np.genfromtxt(csv_path, delimiter=',',skip_header=1,usecols=(4,5,8),dtype=None,encoding='latin-1')
    
    age = np.array(data[:,0])
    height = np.array(data[:,1])
    salary = np.array(data[:,2])

    length = len(data)

    age_train = age[:length-20]
    height_train = height[:length-20]
    salary_train = salary[:length-20]

    age_test = age[length-20:]
    height_test = height[length-20:]
    salary_test = salary[length-20:]

    b0_age,b1_age = simlin_coef(age_train,salary_train)
    b0_height,b1_height = simlin_coef(height_train,salary_train)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    simlin_plot(axes[0], age_train, salary_train, age_test, salary_test, b0_age, b1_age, 'Age')
    simlin_plot(axes[1], height_train, salary_train, height_test, salary_test, b0_height, b1_height, 'Height')

    plt.tight_layout()
    plt.show()
    

def simlin_coef(x,y):
    i = len(x)
    upper = 0
    lower = 0
    for ith in range(i):
        upper += (x[ith]-x.mean())*(y[ith]-y.mean())
        lower += (x[ith]-x.mean())*(x[ith]-x.mean()) #square of the difference between x and mean of x

    b1 = upper/lower
    b0 = np.mean(y) - (b1 * np.mean(x))
    
    return b0, b1

def simlin_plot(ax, x_train, y_train, x_test, y_test, b0, b1, xlabel):

    regression_line = b1 * x_train + b0
    
    ax.plot(x_train, regression_line, c='black')
    ax.scatter(x_train, y_train, c='blue')
    ax.scatter(x_test, y_test, c='red')
   
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Salary')
    ax.set_title('Simple Linear Regression: ' + xlabel + ' vs Salary')
    ax.legend(['Regression Line', 'Train Data', 'Test Data'])

if __name__ == "__main__":
    main()

    
    

