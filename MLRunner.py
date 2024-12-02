import json
import time
import numpy as np

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