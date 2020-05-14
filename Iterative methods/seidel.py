import numpy as np
import scipy.linalg as sl
import time
import matplotlib.pyplot as plt

def diff(answer1, answer2, matrixSize):
    Sum = 0
    for i in range(matrixSize):
        Sum = Sum + (answer1[i] - answer2[i]) ** 2
    Sum = Sum ** 0.5
    return Sum

def solve(A, f, matrixSize, eps):
    newAnswer = [0] * matrixSize
    myAnswer = np.random.rand(matrixSize)
    while diff(myAnswer, newAnswer, matrixSize) > eps:
        myAnswer = newAnswer
        newAnswer = Seidel(A, f, myAnswer, matrixSize)
    return newAnswer

def Seidel(A, f, myAnswer, matrixSize):
    newAnswer = [0] * matrixSize
    for i in range(matrixSize):
        Sum = 0
        for j in range(i - 1):
            Sum = Sum + A[i][j] * newAnswer[j]
        for j in range(i + 1, matrixSize):
            Sum = Sum + A[i][j] * myAnswer[j]
        newAnswer[i] = (f[i] - Sum) / A[i][i]
    return newAnswer

def generate_random_data(matrixSize):
    A = np.random.rand(matrixSize, matrixSize)
    for i in range(matrixSize):
        Sum = 0
        for j in range(matrixSize):
            Sum = Sum + abs(A[i][j])
        A[i][i] = Sum
        f = np.random.rand(matrixSize)
    return A, f

def get_time_data(matrixSize):
    myTime = 0.0
    linalgTime = 0.0
    iterator = 0
    while matrixSize <= 1000:
        A, f = generate_random_data(matrixSize)
        startTime = time.time()
        myAnswer = solve(A, f, matrixSize, eps)
        myTime = time.time() - startTime
        startTime = time.time()
        linalgAnswer = np.linalg.solve(A, f)
        linalgTime = time.time() - startTime
        myTimeData[iterator] = round(myTime, 10)
        linalgTimeData[iterator] = round(linalgTime, 10)
        matrixSize += 10
        iterator += 1
    return myTimeData, linalgTimeData

def show_result(myTimeData, linalgTimeData):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'Red - scipy.linalg.solve(), blue - my implementation',
    	xlabel = 'matrix size (thousand)',
        ylabel = 'time (seconds)')
    ax.plot(myTimeData, color = 'b')
    ax.plot(linalgTimeData, color = 'r')
    plt.savefig('seidel.png', bbox_inches='tight')
    plt.show()

matrixSize = 100
A, f = generate_random_data(matrixSize)

eps = 0.0001
numberOfData = 91
myAnswer = np.zeros(matrixSize)
linalgAnswer = np.zeros(matrixSize)
myTimeData = np.zeros(numberOfData)
linalgTimeData = np.zeros(numberOfData)

myTimeData, linalgTimeData = get_time_data(matrixSize)
show_result(myTimeData, linalgTimeData)