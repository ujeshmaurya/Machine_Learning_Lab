import numpy as np
import math
import time
import sys
import random
import matplotlib.pyplot as  plt

#plotting error curve x axis:no. of iterations y axis: accuracy
def plot_error(ar, x):
	plot_x1 = []
	plot_x2 = []
	for i in range(len(ar)):
		plot_x1.append(i+1)
		plot_x2.append(ar[i]*1)
	plt.plot(plot_x1, plot_x2)
	plt.xlabel("No. of iterations")
	plt.ylabel("Error")
	plt.title(" Plot of Error with iterations")
	plt.ylim(0,200)
	plt.xlim(0, x)
	plt.pause(0.00001)

def sygmoidFunction(x):
	return 1.0/(1+ math.exp(-x))

#numberOfInputs = int(input("Enter the number of inputs to the neural network"))
#numberOfLayers = int(input("Enter the number of hidden layers in the neural network"))
#numberOfNodes = int(input("Enter the number of nodes in each of the layer"))
#learningRate = float(input("Enter the learning Rate!"))
#iterations = int(input("Enter number of iterations in the learning!"))
numberOfInputs=256
numberOfLayers=1
numberOfNodes=12
learningRate=0.5
iterations=10

fileInput = open("test.txt")


NUMBER1 = 9
inputArray = []
outputArray = []
numberOfDataSets = 0
errorArray = []

for eachline in fileInput.readlines():
	readList = list(map(float, eachline.split(" ")))
	tempNum = readList[0]
	readList.pop(0)
	readList=[0.0 if(x == -1) else x for x in readList]
	inputArray.append(readList)
	if(tempNum == NUMBER1):
		outputArray.append(1.0)
	else:
		outputArray.append(0.0)
	numberOfDataSets = numberOfDataSets + 1

weights = []

weights.append([])
for inputNodeIndex in range(numberOfNodes):
	sampleWeights = [random.uniform(-2, 2) for i in range(numberOfInputs)]
	weights[0].append(sampleWeights)

for layerIndex in range(1, numberOfLayers):
	weights.append([])
	for nodeIndex in range(numberOfNodes):
		sampleWeights = [random.uniform(-2, 2) for i in range(numberOfNodes)]
		weights[layerIndex].append(sampleWeights)

weights.append([])
sampleWeights = [random.uniform(-2, 2) for i in range(numberOfNodes)]
weights[numberOfLayers].append(sampleWeights)

for train in range(iterations):
	errorSum = 0.0
	for dataSetIndex in range(len(inputArray)):
		inputData = inputArray[dataSetIndex]
		expectedOutput = outputArray[dataSetIndex]
		forwardData = []
		forwardError = []
		forwardData.append(inputData)
		
		for layerIndex in range(len(weights)):
			layerWeights = weights[layerIndex]
			forwardData.append([])
			forwardError.append([])

			for nodeIndex in range(len(layerWeights)):
				nodeWeights = layerWeights[nodeIndex]
				summation = 0.0
				for i in range(len(inputData)):
					summation = summation + inputData[i] * nodeWeights[i]
				actualOutput = sygmoidFunction(summation)
				forwardData[layerIndex+1].append(actualOutput)

			inputData = forwardData[layerIndex+1]

		actualOutput = inputData[0]
		outputError = (expectedOutput - actualOutput) * (1 - actualOutput) * actualOutput
		errorSum = errorSum + 0.5 * (expectedOutput - actualOutput) * (expectedOutput - actualOutput)
		previousLayerIndex = len(forwardData)-2
		forwardError[previousLayerIndex].append(outputError)

		while previousLayerIndex >= 0:
			for nodeIndex in range(len(weights[previousLayerIndex])):
				for weightIndex in range(len(weights[previousLayerIndex][nodeIndex])):
					weights[previousLayerIndex][nodeIndex][weightIndex] = weights[previousLayerIndex][nodeIndex][weightIndex] + learningRate * forwardError[previousLayerIndex][nodeIndex] * forwardData[previousLayerIndex][weightIndex]
				
			previousLayerIndex = previousLayerIndex - 1
			if(previousLayerIndex >= 0):
				for nodeIndex in range(len(weights[previousLayerIndex])):
					nodeError = 0.0
					for nextErrorIndex in range(len(forwardError[previousLayerIndex + 1])):
						nodeError = nodeError + forwardError[previousLayerIndex + 1][nextErrorIndex] * weights[previousLayerIndex + 1][nextErrorIndex][nodeIndex]
					forwardError[previousLayerIndex].append(forwardData[previousLayerIndex + 1][nodeIndex] * (1 - forwardData[previousLayerIndex+1][nodeIndex]) * nodeError)
	print(errorSum)
	errorArray.append(errorSum)
	plot_error(errorArray, iterations)
#plt.figure(0)

