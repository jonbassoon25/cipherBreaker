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