import joblib

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import MLRunner
import dataCreator

def trainClf(clf, TDTLR, testingPercent = 0.05, dataFormat = "compressed", dtype = "all"):
	print(f"\nCreating data with TD:L of {TDTLR}:1")
	dataCreator.createData(round(TDTLR * (1 + testingPercent)), dtype=dtype)

	#calculation to find testing percent to make currentTDTLR for training data instead of for testing data
	MLRunner.createData(((TDTLR * (1 + testingPercent) - TDTLR) / TDTLR), dataFormat=dataFormat, dataType=dtype)
		
	yTrain, xTrain, yTest, xTest = MLRunner.loadData(dtype)
	#print(xTrain)
	return MLRunner.runML(xTrain, yTrain, xTest, yTest, clf) #also prints training results

#comment out all MLRunner training methods before running
trainingDataToLetterRatio = 131072

clf, name, score, trainingTime = trainClf(DecisionTreeClassifier(criterion="gini", splitter="best"), trainingDataToLetterRatio, 0.05, "compressed", "all")

joblib.dump(clf, "./CCCs/cipherCharacterClassifier.pkl")
with open("./CCCs/cipherClassifierDescription.txt", "w") as CCDFile:
	CCDFile.write(f"{name} trained with {trainingDataToLetterRatio} values per letter\nScore of {round(score,3)}. Training Time of {round(trainingTime,2)}s")