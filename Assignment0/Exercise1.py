import numpy as np
from numpy import random

#question 1 and 2
def question1_2():
    arr = np.zeros(25)
    even = np.array([])
    odd = np.array([])

    for _ in range(25):
        x = random.randint(-100,101)
        np.put(arr,_,x)
        if(x%2 == 0):
            even = np.append(even,x)
        else:
            odd = np.append(odd,x)
    print("The array is: ",arr)
    print("The even numbers are: ",even)
    print("The odd numbers are: ",odd)
    print("The number of even numbers is: ",len(even))
    print("The number of odd numbers is: ",len(odd))
    print("The average of the numbers is: ",np.mean(arr))
    print("The total sum of the numbers is: ",np.sum(arr))

def question3():
    arr = np.zeros((3, 3))
    for i in range(9):
        x = random.rand()
        np.put(arr,i,x)
    print("The matrix \n",arr) # matrix
    print("Second Row" , arr[1,:]) #print the second row
    print("Third column" , arr[:,2]) #print the third column
    print("2nd and 3rd row combined \n",arr[1:3,:]) # print the 2nd and 3rd row combined
    print("1st and 2nd rows; and 2nd and 3rd columns (2x2 matrix) \n",arr[0:2,1:3]) # print the 1st and 2nd rows; and 2nd and 3rd columns (2x2 matrix)
    print("The transpose of c \n",(arr[1:3, :].T)) # print the transpose of the 2nd and 3rd rows

def question4():
    a = np.linspace(1,6,10,endpoint=False)
    b = np.linspace(6,1,10,endpoint=False)

    c = np.array([a,b])
    d = np.array([a,b]).T
    print("The array c is \n",c)
    print("The array d is \n",d)
    x = np.sum(np.square(a - b))
    print("The sum of the squares of the differences between a and b is: ",x)

question1_2()
question3()
question4()


