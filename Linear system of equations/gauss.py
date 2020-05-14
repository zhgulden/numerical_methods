import numpy as np
import time
import scipy.linalg as sl
import matplotlib.pyplot as plt

def forward(A, f, matrixSize):
    for k in range(matrixSize):
        A[k] = A[k] / A[k][k]
        f[k] = f[k] / A[k][k] 
        
        for i in range(k + 1, matrixSize):
            A[i] = A[i] - A[k] * A[i][k]
            f[i] = f[i] - f[k] * A[i][k]
            A[i][k] = 0
    return A, f

def backward(A, f, matrixSize):
    myAnswer = [0] * matrixSize
    for i in range(matrixSize - 1, -1, -1):
        myAnswer[i] = f[i]
        for j in range(i + 1, matrixSize):
            myAnswer[i] = myAnswer[i] - A[i][j] * myAnswer[j]
    return np.array(myAnswer)

def gauss(A, f, matrixSize):
    A, f = forward(A, f, matrixSize)
    myAnswer = backward(A, f, matrixSize)
    return myAnswer

def generate_random_data(matrixSize):
    A = np.random.rand(matrixSize, matrixSize)
    for i in range(n):
        Sum = 0
        for j in range(n):
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
        myAnswer = gauss(A, f, matrixSize)
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
    ax.set(xlim = [0, 90],
        ylim = [0.0, 1.0],
        title = 'Red - scipy.linalg.solve(), blue - gauss()',
        xlabel = 'matrix size (thousand)',
        ylabel = 'time (seconds)')
    ax.plot(myTimeData, color = 'b')
    ax.plot(linalgTimeData, color = 'r')
    plt.savefig('gauss.png', bbox_inches='tight')
    plt.show()

matrixSize = 100

numberOfData = 100
myAnswer = np.zeros(matrixSize)
linalgAnswer = np.zeros(matrixSize)
myTimeData = np.zeros(numberOfData)
linalgTimeData = np.zeros(numberOfData)
myTimeData, linalgTimeData = get_time_data(matrixSize)
show_result(myTimeData, linalgTimeData)
