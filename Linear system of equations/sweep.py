import numpy as np
import scipy.linalg as sl
import time
import matplotlib.pyplot as plt

def sweep(a, b, c, f, n):
    alpha = np.array([0.0] * (n + 1))
    beta = np.array([0.0] * (n + 1))
    for i in range(n):
        alpha[i + 1] = -c[i] / (a[i] * alpha[i] + b[i])
        beta[i + 1] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + b[i])
    x = np.array([0.0] * n)
    x[n - 1] = beta[n]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x
 
def create_matrix(a, b, c):
    a_linalg = np.array([0] + c[:-1].tolist())
    b_linalg = b
    c_linalg = np.array(a[1:].tolist() + [0])
    A = np.array([a_linalg, b_linalg, c_linalg])
    return A

def generate_random_vectors(size):
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.random.rand(size)
    for i in range(size):
        b[i] = abs(a[i]) + abs(b[i]) + abs(c[i]) + 1
    f = np.random.rand(size)
    return a, b, c, f

def get_time_data(a, b, c, f, matrixSize):
    myTime = 0.0
    linalgTime = 0.0
    iterator = 0
    while matrixSize <= 200000:
        a, b, c, f = generate_random_vectors(matrixSize)
        A = create_matrix(a, b, c)
        startTime = time.time()
        myAnswer = sweep(a, b, c, f, matrixSize)
        myTime = time.time() - startTime
        startTime = time.time()
        linalgAnswer = sl.solve_banded((1, 1), A, f)
        linalgTime = time.time() - startTime
        myTimeData[iterator] = round(myTime, 10)
        linalgTimeData[iterator] = round(linalgTime, 10)
        matrixSize += 1000
        iterator += 1
    return myTimeData, linalgTimeData

def show_result(myTimeData, linalgTimeData):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'Red - scipy.linalg.solve_banded(), blue - sweep()',
       xlabel = 'matrix size (thousand)',
       ylabel = 'time (seconds)')
    ax.plot(myTimeData, color = 'b')
    ax.plot(linalgTimeData, color = 'r')
    plt.savefig('sweep.png', bbox_inches='tight')
    plt.show()

matrixSize = 1000
A = np.ones((matrixSize, matrixSize)) * 0.0
a = np.zeros(matrixSize)
b = np.zeros(matrixSize)
c = np.zeros(matrixSize)
f = np.zeros(matrixSize)

numberOfData = 200
myAnswer = np.zeros(matrixSize)
linalgAnswer = np.zeros(matrixSize)
myTimeData = np.zeros(numberOfData)
linalgTimeData = np.zeros(numberOfData)

myTimeData, linalgTimeData = get_time_data(a, b, c, f, matrixSize)
show_result(myTimeData, linalgTimeData)


