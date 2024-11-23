import json
import time
import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import util
import dataCreator
import dataFormatter
import trainingDataCreator


def loadData(dtype = "all"):
	'''
	Loads training data of the given type

	Parameters:
		dtype (str): type of training data to load (lowercase, uppercase, number, symbol, all)
	
	Returns:
		npArray: trainingLabels
		npArray: trainingData
		npArray: testingLabels
		npArray: testingData
	'''
	((abcTrainingLabels, abcTrainingData), 
	 (ABCTrainingLabels, ABCTrainingData), 
	 (numTrainingLabels, numTrainingData), 
	 (symTrainingLabels, symTrainingData)) = util.loadTrainingData()

	((abcTestingLabels, abcTestingData), 
	 (ABCTestingLabels, ABCTestingData), 
	 (numTestingLabels, numTestingData), 
	 (symTestingLabels, symTestingData)) = util.loadTestingData()

	if dtype == "lowercase":
		return abcTrainingLabels, abcTrainingData, abcTestingLabels, abcTestingData
	if dtype == "uppercase":
		return ABCTrainingLabels, ABCTrainingData, ABCTestingLabels, ABCTestingData
	if dtype == "number":
		return numTrainingLabels, numTrainingData, numTestingLabels, numTestingData
	if dtype == "symbol":
		return symTrainingLabels, symTrainingData, symTestingLabels, symTestingData
	if dtype == "all":
		print("Combining Training Data")
		trainingData = np.concatenate((abcTrainingData, ABCTrainingData, numTrainingData, symTrainingData))
		trainingLabels = np.concatenate((abcTrainingLabels, ABCTrainingLabels, numTrainingLabels, symTrainingLabels))
		testingData = np.concatenate((abcTestingData, ABCTestingData, numTestingData, symTestingData))
		testingLabels = np.concatenate((abcTestingLabels, ABCTestingLabels, numTestingLabels, symTestingLabels))
		
		print("Randomizing Training Data")
		trainingLabels, trainingData = trainingDataCreator.randomizeData(trainingLabels, trainingData)
		testingLabels, testingData = trainingDataCreator.randomizeData(testingLabels, testingData)
		return trainingLabels, trainingData, testingLabels, testingData
	

def createData(testingPercent, dataFormat, dataType):
	'''
	Formats data in ./arrays/encp and saves to seperate files

	Parameters:
		testingPercent (float):
		dataFormat (str): how the data should be formatted (uncompressed, compressed)
		dataType (str): type of data to create (lowercase, uppercase, num, sym, all)
	'''
	if dataFormat == "uncompressed":
		dataFormatter.createNumberData()
		trainingDataCreator.createNumberData(testingPercent, dtype=dataType)
	elif dataFormat == "compressed":
		dataFormatter.createCompressedData()
		trainingDataCreator.createCompressedData(testingPercent, dtype=dataType)
	else:
		raise Exception(f"Unknown data format: {dataFormat}")


def saveRun(name, label, score, trainingTime, trainingDataToLetterRatio):
	'''
	Saves a run to the correct classifier results json file and updates the text file to display the new data

	Parameters:
		name (str): classifier name
		label (str): dtype-dataFormat-parameters
		score (float): run score
		trainingTime (float): run training time
		trainingDataToLetterRatio: training data to letter ratio of the run
	'''
	trainingDataToLetterRatio = str(round(trainingDataToLetterRatio))
	trainingTime = round(trainingTime, 2)
	score = round(score, 3)

	#open files
	jsonFile = open(f"./testResults/jsonFiles/{name}.json", "r")
	textFile = open(f"./testResults/textFiles/{name}.txt", "w")

	#load json file and close
	currentRuns = json.load(jsonFile)
	jsonFile.close()

	#add run data to the currentRuns dict
	if label in currentRuns:
		if trainingDataToLetterRatio in currentRuns[label]:
			currentRuns[label][trainingDataToLetterRatio].append([score, trainingTime])
		else:
			currentRuns[label][trainingDataToLetterRatio] = [[score, trainingTime]]
	else:
		currentRuns[label] = {trainingDataToLetterRatio: [[score, trainingTime]]}

	#create new text to write to the text file
	textToWrite = "dtype-dataFormat-params:\n\ttraining values per letter:\n\t\tscore\ttrainingTime\n\n"
	for label in currentRuns.keys():
		textToWrite += f"{label}:\n"
		for trainingDataToLetterRatio in currentRuns[label].keys():
			textToWrite += f"\t{trainingDataToLetterRatio}:\n"
			for trainingResult in currentRuns[label][trainingDataToLetterRatio]:
				tempResult = [str(num) for num in trainingResult]
				textToWrite += f"\t\t{"\t".join(tempResult)}\n"
		textToWrite += "\n\n"
	
	#write to dict to the json file and text to the text file
	jsonFile = open(f"./testResults/jsonFiles/{name}.json", "w")
	json.dump(currentRuns, jsonFile)
	textFile.write(textToWrite.strip())

	#close files
	jsonFile.close()
	textFile.close()


def runML(xTrain, yTrain, xTest, yTest, clf):
	'''
	Runs a ML classifier and prints updates, guesses on testing values, testing values, and the score

	Parameters:
		xTrain (npArray): training data
		yTrain (npArray): training labels
		xTest (npArray): testing data
		yTest (npArray): testing labels
		clf (classifier): ML classifier to use
	
	Returns:
		clf: trained classifier
		str: classifier name
		float: score
		float: training time
	'''
	startTime = time.time()
	score = 0.0
	predictedValues = []

	print(f"\nTraining {type(clf).__name__} with {yTrain.shape[0]} values")
	clf.fit(xTrain, yTrain)
	trainingTime = time.time() - startTime
	print(f"Training completed in {round(trainingTime, 2)} seconds")

	print(f"\nResults against {yTest.shape[0]} values:")
	score = clf.score(xTest, yTest)
	predictedValues = clf.predict(xTest)

	print("  Predictions:\t", predictedValues)
	print("  Actual Labels:", yTest)
	print("  Score:\t", score, end="\n\n")

	return clf, type(clf).__name__, score, trainingTime


def multiRun(trainingDataToLetterRatios, clf, params, trainingReps = 1, dataFormat = "uncompressed", dtype = "all", testingPercent = 0.05):
	'''
	Runs multiple iterations of runML with different training data to letter ratios, saves run data to the appropriate file, and updates the classifier's text file

	Parameters:
		trainingDataToLetterRatios (iterable): iterable of training data to letter ratios to use
		clf (classifier): classifier to train with
		params (str): params of the classifier
		trainingReps (int): training repetitions for each training data to letter ratio
		dataFormat (str): format of data to train on (uncompressed, compressed)
		dtype (str): type of data to train on (lowercase, uppercase, number, symbol, all)
		testingPercent (float): portion of data that should be used for testing (out of 1)
	'''
	for i in range(len(trainingDataToLetterRatios)):
		currentTDTLR = trainingDataToLetterRatios[i]
		for k in range(trainingReps):
			print(f"\nCreating data with TD:L of {currentTDTLR}:1")
			dataCreator.createData(round(currentTDTLR * (1 + testingPercent)), dtype=dtype)

			#calculation to find testing percent to make currentTDTLR for training data instead of for testing data
			createData(((currentTDTLR * (1 + testingPercent) - currentTDTLR) / currentTDTLR), dataFormat, dtype)
		
			yTrain, xTrain, yTest, xTest = loadData(dtype)
			trash, name, score, trainingTime = runML(xTrain, yTrain, xTest, yTest, clf)

			print("Saving Run")
			saveRun(name, f"{dtype}-{dataFormat}-{params}", score, trainingTime, currentTDTLR)


#yTrain, xTrain, yTest, xTest = loadData("all")

#runML(xTrain, yTrain, xTest, yTest, NearestCentroid("euclidean"))
#runML(xTrain, yTrain, xTest, yTest, KNeighborsClassifier(n_neighbors=3, weights = "uniform"))
#runML(xTrain, yTrain, xTest, yTest, GaussianNB())
#runML(xTrain, yTrain, xTest, yTest, DecisionTreeClassifier(criterion="gini", splitter="best"))
#runML(xTrain, yTrain, xTest, yTest, RandomForestClassifier(n_estimators=5))
#runML(xTrain, yTrain, xTest, yTest, SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"))

#max run size < 1048576 before zsh: killed error. Not enough RAM
#multiRun(runs, clf(), "", 1, "compressed")
'''
#short run range (2^6 - 2^11)

runs = [2 ** i for i in range(6, 12)]
multiRun([64], DecisionTreeClassifier(criterion="gini", splitter="best"), "gini,best", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(32, 24), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=10000), "(32,24),relu,lbfgs,1e-5,10000", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=10000), "(100, 100),relu,lbfgs,1e-5,10000", 1, "compressed")
#multiRun(runs, MLPClassifier(hidden_layer_sizes=(192, 140, 100), activation="relu", solver='lbfgs', alpha=1e-5, max_iter=1000000), "(192, 140, 100),relu,lbfgs,1e-5,1000000", 1, "compressed")

'''

'''
#medium run range (2^12 - 2^17)
runs = [2 ** i for i in range(12, 18)]


'''

'''
#log run range (2^18 - 2^20)
runs = [2 ** i for i in range(6, 10)]
multiRun(runs[2:], SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"), "linear,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", 1, "compressed")

runs = [2 ** i for i in range(18, 20)]
multiRun(runs, DecisionTreeClassifier(criterion="gini", splitter="best"), "gini,best", 1, "compressed")
multiRun(runs, DecisionTreeClassifier(criterion="entropy", splitter="best"), "entropy,best", 1, "compressed")
multiRun(runs, DecisionTreeClassifier(criterion="log_loss", splitter="best"), "log_loss,best", 1, "compressed")

multiRun(runs, DecisionTreeClassifier(criterion="gini", splitter="random"), "gini,random", 1, "compressed")
multiRun(runs, DecisionTreeClassifier(criterion="entropy", splitter="random"), "entropy,random", 1, "compressed")
multiRun(runs, DecisionTreeClassifier(criterion="log_loss", splitter="random"), "log_loss,random", 1, "compressed")
'''

'''
#exceptions / long run time sets

#estimated 20hrs per
runs = [2 ** i for i in range(12, 15)] #up to 16384:1
multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")
multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")
multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", trainingReps = 1, dataFormat = "uncompressed", dtype = "all")

#estimated 43hrs per
runs = [2 ** i for i in range(10, 12)] #up to 2048:1
multiRun(runs[1:], SVC(kernel="linear", C=1.0, degree = 3, gamma = "scale"), "linear,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="rbf", C=1.0, degree = 3, gamma = "scale"), "rbf,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="sigmoid", C=1.0, degree = 3, gamma = "scale"), "sigmoid,1.0,3,scale", 1, "compressed")
multiRun(runs, SVC(kernel="poly", C=1.0, degree = 3, gamma = "scale"), "poly,1.0,3,scale", 1, "compressed")
'''


#Decision Tree - gini,best:
#	abc-lor:
#    0.72, 1_000 per letter
#    0.94, 10_000 per letter
#	all:
#    0.47 at all, 1_000 per letter
#    0.78 at all, 10_000 per letter

#Random Forest all-100,gini uses ~25GB Ram for 4096:1 TD:L

#SVC - linear, 1.0
#	abc-lor:
#    0.86 at abc-lor, 1_000 per letter
#    0.95 at abc-lor, 10_000 per letter
#	all:
#    0.79 at all, 10_000 per letter.
