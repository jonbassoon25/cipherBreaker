import matplotlib.pyplot as plt
import math
import util

def compareClassifierParameters(classifierName, params = [], against = "score", dataType = "all"):
	'''
	Graphs impact of different parameters across a constant classifier type

	Parameters:
		classifierName (str): classifier type to compare parameters of
		params (list): list of strings of sections of parameters to compare
		against (str): compare score or time
		dataType (str): data type to compare
	'''
	classifierResults = util.loadTestResultDict(classifierName)

	resultKeys = list(classifierResults.keys())
	data = classifierResults

	#remove all keys that don't match the data type, format, and required params
	for i in range(len(resultKeys) - 1, -1, -1):
		curKey = resultKeys[i]
		curKey = curKey.split("-")
		curKey[1] = f"{curKey[1]}-{curKey.pop(2)}"

		validParams = False
		for j in range(len(params)):
			#print(f"{curKey[1].count(params[j])} of {params[j]} in {curKey[1]}")
			if curKey[1].count(params[j]) > 0:
				validParams = True
				break
		if (not curKey[0] == dataType) or (not len(params) == 0 and not validParams):
			#print("removing " + resultKeys[i])
			resultKeys.pop(i)

		

	#for every key
	plotData = {}
	for resultKey in resultKeys:
		#x values are ratio of data per letter for run
		x = list(data[resultKey].keys())

		#y values are average of the against variable for the given data ratio
		y = []

		for i in range(len(x)):
			runGroup = data[resultKey][x[i]]

			total = 0
			#find average value of the runGroup
			for k in range(len(runGroup)):
				total += runGroup[k][0 if against == "score" else 1]
			total /= len(runGroup)
			#append avg to y
			y.append(total)
		
		plotData[resultKey] = [x, y]
	
	#plot data
	figure, axis = plt.subplots(math.ceil(len(resultKeys)/3), 3)

	highestTime = 0
	for i in range(0, len(resultKeys), 3):
		if len(resultKeys) <= 3:
			for k in range(3):
				#have to have seperate for < 3 because axis array is 1 dimensional
				if k >= len(resultKeys):
					break
				else:
					axis[k].set_title(resultKeys[k])
					axis[k].set_xlabel('Traning Values per Letter')
					axis[k].plot([int(val) for val in plotData[resultKeys[k]][0]], plotData[resultKeys[k]][1])
					axis[k].set_xscale('log')
					axis[k].set_xlim((0, 524288))
					
					if against == "score":
						axis[k].set_ylabel('Score (of 1)')
						axis[k].set_ylim((0, 1))
						axis[k].set_yscale('linear')
					else:
						if plotData[resultKeys[k]][1][-1] > highestTime:
							highestTime = plotData[resultKeys[k]][1][-1]
						axis[k].set_ylabel('Time (seconds)')
						axis[k].set_yscale('log')
			break
		for k in range(3):
			if i + k >= len(resultKeys):
				break
			else:
				#set graph data
				axis[i // 3, k].set_title(resultKeys[i + k])
				axis[i // 3, k].plot([int(val) for val in plotData[resultKeys[i + k]][0]], plotData[resultKeys[i + k]][1])
				axis[i // 3, k].set_xlabel('Traning Values per Letter')
				axis[i // 3, k].set_xscale('log')
				axis[i // 3, k].set_xlim((0, 524288))
				if against == "score":
					axis[i // 3, k].set_ylabel('Score (of 1)')
					axis[i // 3, k].set_ylim((0, 1))
				else:
					axis[i // 3, k].set_ylabel('Time (seconds)')
					
					if not against == "score":
						if plotData[resultKeys[i + k]][1][-1] > highestTime:
							highestTime = plotData[resultKeys[i + k]][1][-1]
						axis[i // 3, k].set_yscale("log")
					else:
						axis[i // 3, k].set_yscale("linear")
	if not against == "score": #against == time
		if len(resultKeys) < 3:
			for k in range(3):
				axis[k].set_ylim((0, math.ceil(highestTime)))
		else:
			for i in range(0, len(resultKeys), 3):
				for k in range(3):
					axis[i // 3, k].set_ylim((0, math.ceil(highestTime)))
	plt.show()

#classifer names passed in as ["DecisionTreeClassifier-gini,random", "GaussianNB-", "SVC-linear,1.0,3,scale"]
def compareClassifiers(classifierNames = [], compare = "score", dataType = "all", dataFormat = "uncompressed"):
	'''
	Graphs different classifiers with specified parameters

	Parameters:
		classifierNames (list): passed in as ["DecisionTreeClassifier-gini,random", "GaussianNB-", "SVC-linear,1.0,3,scale"]
		compare (str): compare score or time
		dataType (str): data type of all classifiers
		dataFormat (str): data format of all classifiers (uncompressed, compressed)
	'''
	allClassifierResults = util.loadAllTestResultDicts()
	
	#split classifierNames and be sure that they are recognized as real classifier names, delete and print message if they aren't
	for i in range(len(classifierNames) - 1, -1, -1):
		classifierNames[i] = classifierNames[i].split("-")
		if not classifierNames[i][0] in allClassifierResults.keys():
			print(f"Classifier not recognized: {classifierNames[i][0]}")
			classifierNames.pop(i)
		
	#if classifierNames length is 0, set classifier names to the keys of allClassifierResults
	if len(classifierNames) == 0:
		for i in range(len(allClassifierResults.keys())):
			classifierNames.append([list(allClassifierResults.keys())[i], list(list(allClassifierResults.values())[i].keys())[0].split("-")[2]])
	
	#create subplots
	figure, axis = plt.subplots(math.ceil(len(classifierNames)/3), 3)

	highestTime = 0
	for i in range(0, len(classifierNames), 3):
		if len(classifierNames) <= 3:
			for k in range(3):
				if k >= len(classifierNames):
					break
				else:
					classifierData = allClassifierResults[classifierNames[k][0]][f"{dataType}-{dataFormat}-{classifierNames[k][1]}"]
					xVals = [int(val) for val in classifierData.keys()]

					#get avg y val for each x val, then push to yVals
					yVals = []
					for vals in classifierData.values():
						curSum = 0
						for val in vals:
							curSum += val[0 if compare == "score" else 1]
						yVals.append(curSum/len(vals))

					axis[k].set_title('-'.join(classifierNames[k]))
					axis[k].set_xlabel("Training Values per Letter")
					axis[k].plot(xVals, yVals)
					axis[k].set_xscale('log')
					axis[k].set_xlim((0, 524288))

					if compare == "score":
						axis[k].set_ylabel('Score (of 1)')
						axis[k].set_ylim((0, 1))
						axis[k].set_yscale('linear')
					else:
						if yVals[-1] > highestTime:
							highestTime = yVals[-1]
						axis[k].set_ylabel('Time (seconds)')
						axis[k].set_yscale('log')
		else:
			for k in range(3):
				if i + k >= len(classifierNames):
					break
				else:
					classifierData = allClassifierResults[classifierNames[i + k][0]][f"{dataType}-{dataFormat}-{classifierNames[i + k][1]}"]
					
					xVals = [int(val) for val in classifierData.keys()]
					#get avg y val for each x val, then push to yVals
					yVals = []
					for vals in classifierData.values():
						curSum = 0
						for val in vals:
							curSum += val[0 if compare == "score" else 1]
						yVals.append(curSum/len(vals))

					#set graph data
					axis[i // 3, k].set_title('-'.join(classifierNames[i + k]))
					axis[i // 3, k].plot(xVals, yVals)

					axis[i // 3, k].set_xlabel('Traning Values per Letter')
					axis[i // 3, k].set_xscale('log')
					axis[i // 3, k].set_xlim((0, 524288))
					
					if compare == "score":
						axis[i // 3, k].set_ylabel('Score (of 1)')
						axis[i // 3, k].set_ylim((0, 1))
						axis[i // 3, k].set_yscale('linear')
					else:
						if yVals[-1] > highestTime:
							highestTime = yVals[-1]
						axis[i // 3, k].set_ylabel('Time (seconds)')
						axis[i // 3, k].set_yscale('log')
					
	if not compare == "score": #against == time
		if len(classifierNames) <= 3:
			for k in range(3):
				axis[k].set_ylim((0, math.ceil(highestTime)))
		else:
			for i in range(0, len(classifierNames), 3):
				for k in range(3):
					axis[i // 3, k].set_ylim((0, math.ceil(highestTime)))
	plt.show()