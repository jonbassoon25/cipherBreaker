import numpy as np
import joblib

import util
import cipherBreaker
import modelAnalyzer

#determine specific failure probability
def determineSFProbability(clfAnalysis, predictedChar, testChar):
	"""
	Determines the probability that the test character is the correct response based on the classifier analysis and its prediction

	Parameters:
		clfAnalysis (dict): dictionary of analysis of relative frequencys of outputs for each input to the clf
		clfPrediction (str): character returned by the classifier model. Has to be an output of the classifier model
		testWord (str): possible correct response. Has to be an output of the classifier model
	
	Returns:
		(float): probability that the test character is correct
	"""
	if not predictedChar in clfAnalysis.keys():
		raise Exception(f"{predictedChar} is not an output of the given clf based on its analysis.")
	if not testChar in clfAnalysis.keys():
		raise Exception(f"{testChar} is not an output of the given clf based on its analysis.")
	
	#result = probability that each letter of test word is correct given the response from the classifier
	
	#let A = the test letter is correct
	#let B = the letter from the classifier response is correct

	#P(B) = probability that the actual letter was translated correctly
	#P(A) = probability that the test char is the actual letter that was translated
	
	#total frequency = frequency true + frequency false
	#P(event) = frequency true / total frequency

	#P(B) = frequency of output of predicted char for predicted char / frequency of incorrect output for predicted char + frequency of correct output of predicted char for predicted char
	#P(A) = frequency of output of predicted char for test char / frequency of correct output of predicted char for predicted char + frequency of incorrect output for predicted char
	#If test letter and predicted letter are the same, P(B) = P(A). True

	#relative frequencys in clfAnalysis can be treated as frequencys because they were all divided by the same number
	
	#freq1 = frequency of output of predicted char for test char
	#freq2 = frequency of correct output of predicted char for predicted char
	#freq3 = frequency of incorrect output for predicted char

	#if key for predicted char in test char does not exist then freq1 = 0 meaning that pA = 0
	if not predictedChar in clfAnalysis[testChar].keys():
		return 0.0
	
	freq1 = clfAnalysis[testChar][predictedChar] #frequency of output of predicted char for test char
	
	freq2 = 0 #frequency of correct output of predicted char for predicted char
	if predictedChar in clfAnalysis[predictedChar]: #if predicted char is not an output for predicted char then freq2 is 0
		freq2 = clfAnalysis[predictedChar][predictedChar] #else it is the frequency of correct output of predicted char for predicted char
	
	freq3 = 0 #frequency of incorrect output for predicted char
	#add all frequencys that aren't outputs of predicted char for predicted char
	for key in clfAnalysis[predictedChar]:
		if not key == predictedChar:
			freq3 += clfAnalysis[predictedChar][key]
	
	
	return round(freq1 / (freq2 + freq3), 4)
	



def predict(clf, outputMessage, trainingType, cutoff = 0.6):
	'''
	Predicts mistranslations of the given output message from a clf

	Parameters:
		clf (clf): classifier used to generate the output message
		outputMessage (str): predicted message returned by the classifier
		trainingType (str): type of data that the classifier was trained on (uncompressed or compressed)
		cutoff (float): minimum confidence of words that will be displayed in predictions
	
	Returns:
		(dict): predicted words for ranges of characters in the output message
	'''
	analysis = modelAnalyzer.analyze(clf, trainingType, 5000, 0.0)

	#can't split on space because space can be mistranslated as another char and another char can be mistranslated as space

	allWords = []
	with open("englishWords.txt", "r") as wordFile:
		allWords = wordFile.readlines()
		allWords = np.array([word.strip() for word in allWords])
	
	#sort from longest to shortest words and record indicies
	wordDict = {}
	for i in range(len(allWords)):
		curWordLength = len(allWords[i])
		if not curWordLength in wordDict.keys():
			#create keys up to this one as well so that they are sorted from least to greatest
			for j in range(list(wordDict.keys())[-1] + 1, curWordLength):
				wordDict[j] = []
			wordDict[curWordLength] = [i]
		else:
			wordDict[curWordLength].append(i)
	#add each section of sorted indicies together
	idx = []
	for value in wordDict.values():
		idx += value

	#assign and reverse to get indexes from greatest to least size
	idx = np.array(idx.reverse())

	#sort allWords from greatest to least with the sorted indicies
	allWords = allWords[idx]
	
		

	
	#steps for mistranslation predictions
	'''
	1) chunk based on output message
	2) let chunk = start to start + length of the longest word in allWords or to end
	2) determine probability that the chunk is one of the words with its length
	3) determine probability that the chunk[:-1] is one of the words with its length all the way down to length 1
	4) find last possible character that could be space in the chunk (character possibly able to translate to space)
	5) repeat steps 3 - 4 with new chunk until no more possible words exist (words are seperated by space)
	6) repeat steps 2 - 5 until end of output message
	7) display results in a user friendly way
	'''

	#find longest word length
	longestWordLength = 0
	for word in allWords:
		if len(word) > longestWordLength:
			longestWordLength = len(word)
	lwl = longestWordLength

	#start chunking
	i = 0
	finished = False
	while not finished:
		chunk = outputMessage[i:i+lwl]
		
	



def predictUserInput():
	pass

trainingType = "uncompressed"
clf = charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")

analysis = modelAnalyzer.analyze(clf, trainingType, 5000)

print(determineSFProbability(analysis, "a", "a"))