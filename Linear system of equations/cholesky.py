import numpy as np
import time
import scipy.linalg as sl
import matplotlib.pyplot as plt

def generate_random_data(matrixSize):
    A = np.random.rand(matrixSize, matrixSize)
    for i in range(matrixSize):
        Sum = 0
        for j in range(matrixSize):
            Sum = Sum + abs(A[i][j])
        A[i][i] = Sum
        f = np.random.rand(matrixSize)
    return A, f

def solve(A, f, matrixSize):
    x = [0] * matrixSize
    for i in range(matrixSize):
        x[i] = f[i]
        for j in range(i):
            x[i] = x[i] - A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return np.array(x)

def cholesky_decomposition(A, matrixSize):
    L = np.ones((matrixSize, matrixSize)) * 0.0
    for i in range(matrixSize):
        for j in range(i + 1):
            if i == j:
                Sum = 0
                for k in range(i):
                    Sum = Sum + L[i][k] ** 2
                L[i][i] = (A[i][i] - Sum) ** 0.5
            else:
                Sum = 0
                for k in range(j):
                    Sum = Sum + L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - Sum) / L[j][j]
    return L

def cholesky_algorithm(A, f, matrixSize):
    L = np.ones((matrixSize, matrixSize)) * 0.0
    L = cholesky_decomposition(A, matrixSize)
    tempAnswer = solve(L, f, matrixSize)
    L.transpose()
    answer = solve(L, tempAnswer, matrixSize)
    return answer

def linalg_algorithm(A, f):
    L = np.linalg.cholesky(A)
    tempAnswer = solve(L, f, matrixSize)
    L.transpose()
    answer = solve(L, tempAnswer, matrixSize)
    return answer

def get_time_data(matrixSize):
    myTime = 0.0
    linalgTime = 0.0
    iterator = 0
    while matrixSize <= 1000:
        A, f = generate_random_data(matrixSize)
        startTime = time.time()
        myAnswer = cholesky_algorithm(A, f, matrixSize)
        myTime = time.time() - startTime
        startTime = time.time()
        linalgAnswer = linalg_algorithm(A, f)
        linalgTime = time.time() - startTime
        myTimeData[iterator] = round(myTime, 10)
        linalgTimeData[iterator] = round(linalgTime, 10)
        matrixSize += 100
        iterator += 1
    return myTimeData, linalgTimeData

def show_result(myTimeData, linalgTimeData):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'Red - linalg cholesky decomposition, blue - my decomposition',
        xlabel = 'matrix size',
        ylabel = 'time (seconds)')
    ax.plot(myTimeData, color = 'b')
    ax.plot(linalgTimeData, color = 'r')
    plt.savefig('cholesky.png', bbox_inches='tight')
    plt.show()

matrixSize = 100
numberOfData = 10
myAnswer = np.zeros(matrixSize)
linalgAnswer = np.zeros(matrixSize)
myTimeData = np.zeros(numberOfData)
linalgTimeData = np.zeros(numberOfData)
myTimeData, linalgTimeData = get_time_data(matrixSize)
show_result(myTimeData, linalgTimeData)


