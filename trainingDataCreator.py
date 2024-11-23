import math
import numpy as np
import cipher
import util

def randomizeData(labels, data):
	'''
	Randomizes the 1st dimension indicies of the given labels and data in the same way

	Parameters:
		labels (npArray): 1 dimensional label array
		data (npArray): multidimensional data array

	Returns:
		npArray: randomized label array
		npArray: randomized data array
	'''
	idx = np.argsort(np.random.random(labels.shape))
	data = data[idx]
	labels = labels[idx]
	return labels, data

def splitData(labels, data, testingPercent):
	'''
	Splits the given labels and data into testing and training sets

	Parameters:
		labels (npArray): 1 dimensional label array
		data (npArray): multidimensional data array
		testingPercent (float): portion of data that should be used for testing (out of 1)

	Returns:
		npArray: testing labels
		npArray: testing data
		npArray: training labels
		npArray: training data
	'''
	splitNum = math.floor(testingPercent * labels.shape[0])
	testingLabels = labels[:splitNum]
	testingData = data[:splitNum]
	trainingLabels = labels[splitNum:]
	trainingData = data[splitNum:]
	return testingLabels, testingData, trainingLabels, trainingData

def generateLabels(dtype, data):
	'''
	Generates labels of the given data type for the given data

	Parameters:
		dtype (str): data type for label generation (lowercase, uppercase, num, sym, all)
		data (npArray): data array to generate labels for
	
	Returns:
		npArray: 1 dimensional label array with 1 value for each value in the given data array
	'''
	if dtype == "lowercase":
		labelChars = cipher.abc
	elif dtype == "uppercase":
		labelChars = cipher.ABC
	elif dtype == "num":
		labelChars = cipher.str_num
	elif dtype == "sym":
		labelChars = cipher.sym
	elif dtype == "all":
		labelChars = cipher.abc + cipher.ABC + cipher.str_num + cipher.sym
	else:
		raise Exception(f"Data type {dtype} not recognized.")
	
	return np.array([[letter for i in range(data.shape[1])] for letter in labelChars]).ravel()

def createCompressedData(testingPercent = 0.05, dtype = "all"):
	'''
	Creates and saves compressed testing and training data to folders in ./arrays

	Parameters:
		testingPercent (float): portion of data that should be used for testing (out of 1)
		dtype (str): type of data to encrypt (lowercase, uppercase, num, sym, all)
	'''
	com_abc, com_ABC, com_num, com_sym = util.loadComData()

	if dtype == "lowercase" or dtype == "all":
		labels = generateLabels("lowercase", com_abc)
		data = com_abc.ravel()
		data = data.reshape(len(data) // 36, 36)

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/abc_testing_labels-lor", testingLabels)
		util.saveArray("testingData/abc_testing_data-lor", testingData)
		util.saveArray("trainingLabels/abc_training_labels-lor", trainingLabels)
		util.saveArray("trainingData/abc_training_data-lor", trainingData)

	if dtype == "uppercase" or dtype == "all":
		labels = generateLabels("uppercase", com_ABC)
		data = com_ABC.ravel()
		data = data.reshape(len(data) // 36, 36)

		labels, data = randomizeData(labels, data)

		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/ABC_testing_labels-upr", testingLabels)
		util.saveArray("testingData/ABC_testing_data-upr", testingData)
		util.saveArray("trainingLabels/ABC_training_labels-upr", trainingLabels)
		util.saveArray("trainingData/ABC_training_data-upr", trainingData)

	if dtype == "num" or dtype == "all":
		labels = generateLabels("num", com_num)
		data = com_num.ravel()
		data = data.reshape(len(data) // 36, 36)

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/num_testing_labels-num", testingLabels)
		util.saveArray("testingData/num_testing_data-num", testingData)
		util.saveArray("trainingLabels/num_training_labels-num", trainingLabels)
		util.saveArray("trainingData/num_training_data-num", trainingData)

	if dtype == "sym" or dtype == "all":
		labels = generateLabels("sym", com_sym)
		data = com_sym.ravel()
		data = data.reshape(len(data) // 36, 36)

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/sym_testing_labels-sym", testingLabels)
		util.saveArray("testingData/sym_testing_data-sym", testingData)
		util.saveArray("trainingLabels/sym_training_labels-sym", trainingLabels)
		util.saveArray("trainingData/sym_training_data-sym", trainingData)
		
	
def createNumberData(testingPercent = 0.05, dtype = "all"):
	'''
	Creates and saves compressed testing and training data to folders in ./arrays

	Parameters:
		testingPercent (float): portion of data that should be used for testing (out of 1)
		dtype (str): type of data to encrypt (lowercase, uppercase, num, sym, or all)
	'''
	num_abc, num_ABC, num_num, num_sym = util.loadNumData()

	if dtype == "lowercase" or dtype == "all":
		labels = np.array([[letter for i in range(num_abc.shape[1])] for letter in cipher.abc]).ravel()
		data = np.array(num_abc).ravel()
		data = np.array([[int(num) for num in nums] for nums in data])

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/abc_testing_labels-lor", testingLabels)
		util.saveArray("testingData/abc_testing_data-lor", testingData)
		util.saveArray("trainingLabels/abc_training_labels-lor", trainingLabels)
		util.saveArray("trainingData/abc_training_data-lor", trainingData)

	if dtype == "uppercase" or dtype == "all":
		labels = np.array([[letter for i in range(num_ABC.shape[1])] for letter in cipher.ABC]).ravel()
		data = np.array(num_ABC).ravel()
		data = np.array([[int(num) for num in nums] for nums in data])

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/ABC_testing_labels-upr", testingLabels)
		util.saveArray("testingData/ABC_testing_data-upr", testingData)
		util.saveArray("trainingLabels/ABC_training_labels-upr", trainingLabels)
		util.saveArray("trainingData/ABC_training_data-upr", trainingData)

	if dtype == "num" or dtype == "all":
		labels = np.array([[letter for i in range(num_num.shape[1])] for letter in cipher.str_num]).ravel()
		data = np.array(num_num).ravel() #1d array of strings of nums
		data = np.array([[int(num) for num in nums] for nums in data]) #2d array of nums

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/num_testing_labels-num", testingLabels)
		util.saveArray("testingData/num_testing_data-num", testingData)
		util.saveArray("trainingLabels/num_training_labels-num", trainingLabels)
		util.saveArray("trainingData/num_training_data-num", trainingData)

	if dtype == "sym" or dtype == "all":
		labels = np.array([[letter for i in range(num_sym.shape[1])] for letter in cipher.sym]).ravel()
		data = np.array(num_sym).ravel()

		data = np.array([[int(num) for num in nums] for nums in data])

		labels, data = randomizeData(labels, data)
		testingLabels, testingData, trainingLabels, trainingData = splitData(labels, data, testingPercent)

		util.saveArray("testingLabels/sym_testing_labels-sym", testingLabels)
		util.saveArray("testingData/sym_testing_data-sym", testingData)
		util.saveArray("trainingLabels/sym_training_labels-sym", trainingLabels)
		util.saveArray("trainingData/sym_training_data-sym", trainingData)