import numpy as np

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

def find_index(array, value):
    eps = 1
    left, right = 0, len(array) - 1
    while right - left > eps:
        middle = (left + right) // 2
        if array[middle] >= value:
            right = middle
        else:
            left = middle
    return left

def generate_smooth_grid(x, y):
    n = len(x) - 1 
    h = (x[n] - x[0]) / n
    a = np.array([0] + [1] * (n - 1) + [0])
    b = np.array([1] + [4] * (n - 1) + [1])
    c = np.array([0] + [1] * (n - 1) + [0])
    f = np.zeros(n + 1)
    for i in range(1, n):
        f[i] = 3 * (y[i - 1] - 2 * y[i] + y[i + 1]) / h ** 2
    s = sweep(a, b, c, f, n + 1)
    A = np.array([0.0] * (n + 1))
    B = np.array([0.0] * (n + 1))
    C = np.array([0.0] * (n + 1))
    D = np.array([0.0] * (n + 1))
    for i in range(n):
        D[i] = y[i]
        B[i] = s[i]
        A[i] = (B[i + 1] - B[i]) / (3 * h)
        if i != n - 1:
            C[i] = (y[i + 1] - y[i]) / h - (B[i + 1] + 2 * B[i]) * h / 3
        else:
            C[i] = (y[i + 1] - y[i]) / h - (2 * B[i]) * h / 3
    return A, B, C, D

def open_files(filename1, filename2, filename3, filename4):
    trainX = open(filename1, 'r')
    trainY = open(filename2, 'r')
    testX = open(filename3, 'r')
    testY = open(filename4, 'w')
    return trainX, trainY, testX, testY

def close_files(trainX, trainY, testX, testY):
    trainX.close()
    trainY.close()
    testX.close()
    testY.close()

def get_number_of_data(descriptor):
    numberOfData = int(descriptor.readline())
    return numberOfData

def get_data(descriptor):
    data = [float(i) for i in descriptor.readline().split()]
    return data

trainX, trainY, testX, testY = open_files('train.dat', 'train.ans', 'test.dat', 'test.ans')
numTrainDat, numTestDat = get_number_of_data(trainX),  get_number_of_data(testX)
trainY.readline()
dataTrainX, dataTrainY, dataTestX = get_data(trainX), get_data(trainY), get_data(testX)

A, B, C, D = generate_smooth_grid(dataTrainX, dataTrainY)

#inputX = open('train.dat', 'r')
#inputY = open('train.ans', 'r')
#inputFindX = open('test.dat', 'r')
#outputY = open('test.ans', 'w')

#N = int(inputX.readline())
#inputY.readline()
#M = int(inputFindX.readline())

#x = np.array([float(i) for i in inputX.readline().split()])
#y = np.array([float(i) for i in inputY.readline().split()])
#newX = np.array([float(i) for i in inputFindX.readline().split()])

testY.write(str(numTrainDat) + '\n')
for i in range(numTestDat):
    index = find_index(dataTrainX, dataTestX[i])
    tmp = dataTestX[i] - dataTrainX[index]
    value = A[index] * (tmp ** 3) + B[index] * (tmp ** 2) + C[index] * tmp + D[index]
    testY.write(str(value) + ' ')

close_files(trainX, trainY, testX, testY)