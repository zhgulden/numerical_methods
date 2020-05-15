import numpy as np
import matplotlib.pyplot as plt

def find_index(array, value):
    left, right = 0, len(array) - 1
    while right - left > 1:
        middle = (left + right) // 2
        if array[middle] >= value:
            right = middle
        else:
            left = middle
    return left

def build_segment(x, y):
    N = len(x)
    k, b = [0.0] * (N - 1), [0.0] * (N - 1)
    for i in range(N - 1):
        tmp = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        k[i] = tmp
        b[i] = y[i] - x[i] * tmp
    return k, b

def open_files(filename1, filename2, filename3, filename4):
    trainX = open(filename1, 'r')
    trainY = open(filename2, 'r')
    testX = open(filename3, 'r')
    testY = open(filename4, 'w')
    return trainX, trainY, testX, testY

def get_number_of_data(descriptor):
    numberOfData = int(descriptor.readline())
    return numberOfData

def get_data(descriptor):
    data = [float(i) for i in descriptor.readline().split()]
    return data

trainX, trainY, testX, testY = open_files('train.dat', 'train.ans', 'test.dat', 'test.ans')
numTrainDat, numTestDat = get_number_of_data(trainX),  get_number_of_data(testX)

trainY.readline()

dataTrainX = get_data(trainX)
dataTrainY = get_data(trainY)
dataTestX = get_data(testX)

a, b = build_segment(dataTrainX, dataTrainY)

testY.write(str(numTrainDat) + '\n')
for i in range(numTestDat):
    index = find_index(dataTrainX, dataTestX[i])
    value = a[index] * dataTestX[i] + b[index]
    testY.write(str(value) + ' ')
trainX.close()
trainY.close()
testX.close()
testY.close()

