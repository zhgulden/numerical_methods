import numpy as np
import matplotlib.pyplot as plt

def Lagrange(dataTrainX, dataTrainY, dataTestX):
    value = 0.0
    n = len(dataTrainX)
    for i in range(n):
        if dataTestX == dataTrainX[i]:
            return dataTrainY[i]
    for i in range(n):
        tmp = 1.0
        for j in range(n):
            if i != j:
                tmp = (tmp * (dataTestX - dataTrainX[j])) / (dataTrainX[i] - dataTrainX[j])
        value = value + dataTrainY[i] * tmp     
    return value

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
dataTrainX, dataTrainY, dataTestX = get_data(trainX), get_data(trainY), get_data(testX)

testY.write(str(numTrainDat) + '\n')
for i in range(numTestDat):
    value = Lagrange(dataTrainX, dataTrainY, dataTestX[i])
    testY.write(str(value) + ' ')

trainX.close()
trainY.close()
testX.close()
testY.close()