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
    n = len(x)
    a, b = [0.0] * (n - 1), [0.0] * (n - 1)
    for i in range(n - 1):
        tmp = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        a[i] = tmp
        b[i] = y[i] - x[i] * tmp
    return a, b

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

def show_result(dataTrainX, dataTrainY, dataTestX, dataTestY):
    plt.figure()
    plt.title('Linear interpolation')
    plt.subplot(2, 1, 1)
    plt.plot(dataTrainX, dataTrainY, 'r')
    plt.subplot(2, 1, 2)
    plt.plot(dataTestX, dataTestY, 'b')
    plt.savefig('Linear.png', bbox_inches='tight')
    plt.show()

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

testY = open('test.ans', 'r')
testY.readline()
dataTestY = get_data(testY)
print(dataTestY)

trainX.close()
trainY.close()
testX.close()
testY.close()

show_result(dataTrainX, dataTrainY, dataTestX, dataTestY)
