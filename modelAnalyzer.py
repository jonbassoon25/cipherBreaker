import numpy as np
import matplotlib.pyplot as plt
import joblib

import cipher
import util
import cipherBreaker

#analizes the current cipher breaker model and produces a series of pie charts that show what results it produces for each input

def analyze(clf, trainingType, numTests, cutoff = 0.05):
	analysis = {}
	testingLetters = cipher.abc + cipher.ABC + cipher.str_num + cipher.sym
	for letter in testingLetters:
		testingData = np.array([cipher.to_cipher(letter) for j in range(numTests)])

		if trainingType == "uncompressed":
			testingData = cipherBreaker.formatAsNumbers(testingData)
		elif trainingType == "compressed":
			testingData = cipherBreaker.formatAsCompressed(np.array([testingData]))[0]
		
		result = clf.predict(testingData)
		
		#find frequencys of each result character
		frequencys = {}
		for i in range(len(result)):
			if not result[i] in frequencys.keys():
				frequencys[result[i]] = 1
			else:
				frequencys[result[i]] += 1
		#convert frequencys into relative frequencys
		percentRelFrequencys = {}
		for frequencyKey in frequencys.keys():
			percentRelFrequency = round(frequencys[frequencyKey] / len(result) * 100, 2)
			if percentRelFrequency < cutoff * 100:
				if "other" in percentRelFrequencys.keys():
					percentRelFrequencys["other"] += percentRelFrequency
				else:
					percentRelFrequencys["other"] = percentRelFrequency
			else:
				percentRelFrequencys[frequencyKey] = percentRelFrequency

		#store the percentRelFrequency of this character in analysis
		analysis[letter] = util.sortDict(percentRelFrequencys)
	return analysis

def plotAnalysis(analysis):
	#analysis should have keys = cipher.abc + cipher.ABC + cipher.str_num + cipher.sym

	#abc-lor
	for k in range(0, 26, 13):
		fig, axis = plt.subplots(5,3)
		for i in range(0, 13, 3):
			for j in range(3):
				if i + j >= 13:
					break
				sizes = list(analysis[cipher.abc[k + i + j]].values())
				labels = list(analysis[cipher.abc[k + i + j]].keys())
				axis[i // 3][j].pie(sizes, labels=labels, autopct='%1.1f%%')
				axis[i // 3][j].set_title(f"Relative Frequency of outputs given {cipher.abc[k + i + j]}")
	
	#ABC-upr
	for k in range(0, 26, 13):
		fig, axis = plt.subplots(5,3)
		for i in range(0, 13, 3):
			for j in range(3):
				if i + j >= 13:
					break
				sizes = list(analysis[cipher.ABC[k + i + j]].values())
				labels = list(analysis[cipher.ABC[k + i + j]].keys())
				axis[i // 3][j].pie(sizes, labels=labels, autopct='%1.1f%%')
				axis[i // 3][j].set_title(f"Relative Frequency of outputs given {cipher.ABC[k + i + j]}")
	
	#num-num
	fig, axis = plt.subplots(5,2)
	for i in range(0, 10, 2):
		for j in range(2):
			if i + j >= len(cipher.str_num):
				break
			sizes = list(analysis[cipher.str_num[i + j]].values())
			labels = list(analysis[cipher.str_num[i + j]].keys())
			axis[i // 2][j].pie(sizes, labels=labels, autopct='%1.1f%%')
			axis[i // 2][j].set_title(f"Relative Frequency of outputs given {cipher.str_num[i + j]}")

	#abc-lor
	for k in range(0, 33, 11):
		fig, axis = plt.subplots(4, 3)
		for i in range(0, 11, 3):
			for j in range(3):
				if i + j >= 11:
					break
				sizes = list(analysis[cipher.sym[k + i + j]].values())
				labels = list(analysis[cipher.sym[k + i + j]].keys())
				axis[i // 3][j].pie(sizes, labels=labels, autopct='%1.1f%%')
				axis[i // 3][j].set_title(f"Relative Frequency of outputs given {cipher.sym[k + i + j]}")


	plt.show()




			

numTests = 5000 #number of times each character is tested for accuracy
trainingType = "uncompressed"
#charClassifier = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")

analysis = analyze(charClassifier, trainingType, numTests)
plotAnalysis(analysis)

