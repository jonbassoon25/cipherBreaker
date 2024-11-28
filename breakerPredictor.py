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
	
def determineWordProbability(clfAnalysis, predictedWord, testWord):
	if not len(predictedWord) == len(testWord):
		raise Exception("Length of predicted word and test word do not match.")
	
	probability = 1 #probability that the testWord is the correct word
	for i in range(len(predictedWord)):
		probability *= determineSFProbability(clfAnalysis, predictedWord[i], testWord[i]) #series of and = series of multiplication
		
		if probability == 0.0: #if the probability is 0, then stop
			break
	
	return round(probability, 4)



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
	print("Analyzing Model...")
	analysis = modelAnalyzer.analyze(clf, trainingType, 5000, 0.0)

	#can't split on space because space can be mistranslated as another char and another char can be mistranslated as space

	print("Loading Words...")
	allWords = []
	with open("words.txt", "r") as wordFile:
		allWords = wordFile.readlines()
		allWords = np.array([word.strip() for word in allWords])
	
	#sort from longest to shortest words and record indicies
	wordDict = {0:[]}
	for i in range(len(allWords)):
		curWordLength = len(allWords[i])
		if not curWordLength in wordDict.keys():
			#create keys up to this one as well so that they are sorted from least to greatest
			for j in range(list(wordDict.keys())[-1] + 1, curWordLength):
				wordDict[j] = []
			wordDict[curWordLength] = [i]
		else:
			wordDict[curWordLength].append(i)
	#convert indicies in wordDict to actual values from allWords
	for key in wordDict.keys():
		for i in range(len(wordDict[key])):
			wordDict[key][i] = allWords[wordDict[key][i]]
		#delete keys without words in them and make all other keys np arrays
		wordDict[key] = np.array(wordDict[key])
	
	for i in range(len(wordDict.keys()) - 1, -1, -1):
		if len(list(wordDict.values())[i]) == 0:
			del wordDict[list(wordDict.keys())[i]]
	
	#steps for mistranslation predictions
	'''
	1) chunk based on output message
	2) let chunk = start to start + length of the longest word in allWords or to end
	3) determine probability that the chunk is one of the words with its length
	4) find last possible character that could be space in the chunk (character possibly able to translate to space)
	5) repeat steps 3 - 4 with new chunk until no more possible words exist (words are seperated by space)
	6) remove the smallest chunk from the output message and repeat steps 2 - 5 until end of output message
	7) display results in a user friendly way
	'''

	#find longest word length
	lwl = list(wordDict.keys())[-1]

	possibleWords = {}

	#start chunking
	print("Determining Possible Words...")
	i = 0
	while i < len(outputMessage):
		if i + lwl < len(outputMessage):
			chunk = outputMessage[i:i+lwl]
		else:
			chunk = outputMessage[i:]

		#print(chunk)

		#find possible space indicies
		possibleSpaceIndicies = []
		for j in range(len(chunk)):
			if not determineSFProbability(analysis, chunk[j], " ") == 0.0:
				possibleSpaceIndicies.append(j)
		possibleSpaceIndicies.append(lwl) #include possible word from last space to the end of the chunk
		possibleSpaceIndicies.reverse()
		
		#loop through each subchunk of the chunk
		for possibleIndex in possibleSpaceIndicies:
			subchunk = chunk[:possibleIndex]
			#print(subchunk)
			#calculate probability of each word of the length of the subchunk
			if not len(subchunk) in wordDict.keys():
				continue #no words of length subchunk
			for word in wordDict[len(subchunk)]:
				wordProbability = determineWordProbability(analysis, subchunk, word)
				if wordProbability > 0.0: #word is possible
					print(word, wordProbability)
					chunkRange = f"{i}-{i + possibleIndex}"
					#assign word to a range in possible words
					if not chunkRange in possibleWords:
						possibleWords[chunkRange] = [word, wordProbability]
					else:
						possibleWords[chunkRange].append([word, wordProbability])
			#find next subchunk by continuing loop
		#remove first subchunk from next iteration
		print(possibleSpaceIndicies)
		i += possibleSpaceIndicies[-1] + 1
	
	print(possibleWords)


	



def predictUserInput():
	pass

trainingType = "uncompressed"
clf = charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")

#analysis = modelAnalyzer.analyze(clf, trainingType, 5000)

#print(determineSFProbability(analysis, "a", "a"))

predict(clf, "my hovercraft is full of eels", trainingType, cutoff = 0.6)