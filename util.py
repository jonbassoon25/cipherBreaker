import glob
import json
import numpy as np
import cipher

def saveArray(name, array):
	'''
	Saves the given array to ./arrays/name.npy

	Parameters:
		name (str): path/filename to store array at
		array (npArray): array to store
	'''
	np.save(f"./arrays/{name}.npy", array)

def loadArray(name):
	'''
	Loads an array from ./arrays/name.npy

	Parameters:
		name (str): path/filename to load array from
	
	Returns:
		npArray: loaded array
	'''
	return np.load(f"./arrays/{name}.npy")

def loadEncpData():
	'''
	Loads encrypted data arrays from ./arrays/encp and converts them into 2d arrays

	Returns:
		npArray: encrypted lowercase letters
		npArray: encrypted uppercase letters
		npArray: encrypted numbers
		npArray: encrypted symbols
	'''
	encp_abc = np.array([loadArray(f"encp/{char}-lor") for char in cipher.abc])
	encp_ABC = np.array([loadArray(f"encp/{char}-upr") for char in cipher.ABC])
	encp_num = np.array([loadArray(f"encp/{char}-num") for char in cipher.str_num])
	encp_sym = np.array([loadArray(f"encp/{char}-sym") for char in cipher.sym])
	return encp_abc, encp_ABC, encp_num, encp_sym

def loadComData():
	'''
	Loads compressed data arrays from ./arrays/com_encp

	Returns:
		npArray: compressed lowercase letters
		npArray: compressed uppercase letters
		npArray: compressed numbers
		npArray: compressed symbols
	'''
	com_abc = np.array(loadArray("com_encp/abc-lor"))
	com_ABC = np.array(loadArray("com_encp/ABC-upr"))
	com_num = np.array(loadArray("com_encp/num-num"))
	com_sym = np.array(loadArray("com_encp/sym-sym"))
	return com_abc, com_ABC, com_num, com_sym

def loadNumData():
	'''
	Loads uncompressed data arrays from ./arrays/num_encp

	Returns:
		npArray: uncompressed lowercase letters
		npArray: uncompressed uppercase letters
		npArray: uncompressed numbers
		npArray: uncompressed symbols
	'''
	num_abc = np.array(loadArray("num_encp/abc-lor"))
	num_ABC = np.array(loadArray("num_encp/ABC-upr"))
	num_num = np.array(loadArray("num_encp/num-num"))
	num_sym = np.array(loadArray("num_encp/sym-sym"))
	return num_abc, num_ABC, num_num, num_sym

def loadTrainingData():
	'''
	Loads training label and data arrays from ./arrays/trainingLabels and ./arrays/trainingData

	Returns:
		(npArray, npArray): lowercase training labels, lowercase training data
		(npArray, npArray): uppercase training labels, uppercase training data
		(npArray, npArray): number training labels, number training data
		(npArray, npArray): symbol training labels, symbol training data
	'''
	abcTrainingLabels = loadArray("trainingLabels/abc_training_labels-lor")
	ABCTrainingLabels = loadArray("trainingLabels/ABC_training_labels-upr")
	numTrainingLabels = loadArray("trainingLabels/num_training_labels-num")
	symTrainingLabels = loadArray("trainingLabels/sym_training_labels-sym")

	abcTrainingData = loadArray("trainingData/abc_training_data-lor")
	ABCTrainingData = loadArray("trainingData/ABC_training_data-upr")
	numTrainingData = loadArray("trainingData/num_training_data-num")
	symTrainingData = loadArray("trainingData/sym_training_data-sym")

	return (abcTrainingLabels, abcTrainingData), (ABCTrainingLabels, ABCTrainingData), (numTrainingLabels, numTrainingData), (symTrainingLabels, symTrainingData)

def loadTestingData():
	'''
	Loads testing label and data arrays from ./arrays/testingLabels and ./arrays/testingData

	Returns:
		(npArray, npArray): lowercase testing labels, lowercase testing data
		(npArray, npArray): uppercase testing labels, uppercase testing data
		(npArray, npArray): number testing labels, number testing data
		(npArray, npArray): symbol testing labels, symbol testing data
	'''
	abcTestingLabels = loadArray("testingLabels/abc_testing_labels-lor")
	ABCTestingLabels = loadArray("testingLabels/ABC_testing_labels-upr")
	numTestingLabels = loadArray("testingLabels/num_testing_labels-num")
	symTestingLabels = loadArray("testingLabels/sym_testing_labels-sym")

	abcTestingData = loadArray("testingData/abc_testing_data-lor")
	ABCTestingData = loadArray("testingData/ABC_testing_data-upr")
	numTestingData = loadArray("testingData/num_testing_data-num")
	symTestingData = loadArray("testingData/sym_testing_data-sym")

	return (abcTestingLabels, abcTestingData), (ABCTestingLabels, ABCTestingData), (numTestingLabels, numTestingData), (symTestingLabels, symTestingData)

def loadTestResultDict(type):
	'''
	Loads the specifid test result dictionary from ./testResults/jsonFiles/type.json

	Parameters:
		type (str): test result type to load (name of learning alg)
	
	Returns:
		dict: test results of the given type of learning alg
	'''
	with open(f"./testResults/jsonFiles/{type}.json") as jsonFile:
		return json.load(jsonFile)

def loadAllTestResultDicts():
	'''
	Loads all known test results from ./testResults/jsonFiles

	Returns:
		dict: all test results of learning alg tests
	'''
	testData = {}
	#find all paths to classifer test result json files
	classifiers = glob.glob("./testResults/jsonFiles/*.json")
	#translate paths into classifier names
	for i in range(len(classifiers)):
		classifiers[i] = classifiers[i][len("./testResults/jsonFiles/"):-len(".json")]
	#load test data for each classifier
	for classifier in classifiers:
		testData[classifier] = loadTestResultDict(classifier)
	return testData