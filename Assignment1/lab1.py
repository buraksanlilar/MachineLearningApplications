import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path(__file__).resolve().parents[1] / 'Football_players.csv'
    data = np.genfromtxt(csv_path, delimiter=',',skip_header=1,usecols=(4,8),dtype=None,encoding='latin-1')
    #age is x and salary is y
    x = np.array(data[:,0])
    y = np.array(data[:,1])

    b0,b1 = simlin_coef(x,y)
    simlin_plot(x,y,b0,b1)


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

def simlin_plot(x,y,b0,b1):
    regression_line = b1*x + b0
    plt.scatter(x,y,c='blue')
    plt.plot(x,regression_line,c='red')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.title('Simple Linear Regression: Age vs Salary')

    plt.show()

if __name__ == "__main__":
    main()