import numpy as np
import joblib

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
	
def determineStringProbability(clfAnalysis, predictedString, testString):
	if not len(predictedString) == len(testString):
		return 0.0 #if predicted string and test string don't match length, testString cannot be correct
	
	probability = 1 #probability that the testWord is the correct word
	for i in range(len(predictedString)):
		probability *= determineSFProbability(clfAnalysis, predictedString[i], testString[i]) #series of and = series of multiplication
		
		if probability == 0.0: #if the probability is 0, then stop
			break
	
	return probability


#cannot predict most words that are followed by punctuation
def predict(clf, outputMessage, trainingType, cutoff = 0.4, minOutput = 0, maxOutput = -1, includePredictedTextAsWords = False):
	'''
	Predicts mistranslations of the given output message from a clf

	Parameters:
		clf (clf): classifier used to generate the output message
		outputMessage (str): predicted message returned by the classifier
		trainingType (str): type of data that the classifier was trained on (uncompressed or compressed)
		cutoff (float): minimum confidence of words that will be displayed in predictions
		minOutput (int): minimum number of predicted messages to output, overrides probability cutoff
		maxOutput (int): maximum number of predicted messages to output, overrides probability cutoff (ignored if -1)
	
	Returns:
		(dict): predicted words for ranges of characters in the output message
	'''
	if minOutput > maxOutput and not maxOutput == -1:
		raise Warning(f"Maximum output of {maxOutput} less than minimum output of {minOutput}. Setting maximum output to minimum output.")
		maxOutput = minOutput
	
	print("Analyzing Model...")
	analysis = modelAnalyzer.analyze(clf, trainingType, 5000, 0.0)

	#can't split on space because space can be mistranslated as another char and another char can be mistranslated as space

	print("Loading Words...")
	allWords = []
	with open("words.txt", "r") as wordFile:
		allWords = wordFile.readlines()
		allWords = np.array([[word.strip().lower(), word.strip().upper(), word.strip().title()] for word in allWords]).ravel()
	
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
				wordProbability = round(determineStringProbability(analysis, subchunk, word), 3) #round so that many words that will have a probabiliy of near 0 aren't included
				if wordProbability > 0.0: #word is possible
					#print(word, wordProbability)
					#assign word to a range in possible words
					if not i in possibleWords:
						possibleWords[i] = [str(word)]
					else:
						possibleWords[i].append(str(word))
			#find next subchunk by continuing loop
		#remove first subchunk from next iteration
		i += possibleSpaceIndicies[-1] + 1
	
	#each word of the output message is also a possible word (broken apart by spaces)
	if includePredictedTextAsWords:
		splitMessage = outputMessage.split(" ")
		curLen = 0
		for i in range(len(splitMessage)):
			word = splitMessage[i]

			if not curLen in possibleWords:
				possibleWords[curLen] = [word]
			elif not word in possibleWords[curLen]: #don't add the word if it is already in the possible words
				possibleWords[curLen].append(word)
			curLen += len(word) + 1

	totalWords = 0
	for key in possibleWords:
		totalWords += len(possibleWords[key])
	print(f"Determining Possible Messages from {totalWords} Possible Words...")
	
	possibleMessages = [possibleWords[0][i] + " " for i in range(len(possibleWords[0]))] #possible messages always start from beginning (index 0)
	addingWords = True
	while addingWords:
		#add to the end of each possible message if able to
		oldPossibleMessages = possibleMessages[:]
		possibleMessages = []
		for i in range(len(oldPossibleMessages)):
			if not len(oldPossibleMessages[i]) in possibleWords.keys(): #no new words for this message
				possibleMessages.append(oldPossibleMessages[i])
				addingWords = False
				continue
			#there are new words for this message
			addingWords = True
			newWords = possibleWords[len(oldPossibleMessages[i])]
			for j in range(len(newWords)):
				possibleMessages.append(oldPossibleMessages[i] + newWords[j] + " ") #add a new message ending with a new word and space
	
	#remove space from end of all possible messages
	possibleMessages = [message[:-1] for message in possibleMessages]

	print("Scoring and Sorting Possible Messages...")

	#remove duplicates in possible messages
	possibleMessages = list(dict.fromkeys(possibleMessages))

	#score possible messages
	possibleMessageScores = [determineStringProbability(analysis, outputMessage, possibleMessages[i]) for i in range(len(possibleMessages))]

	#sort scores and messages from highest to lowest
	idx = [i for i in range(len(possibleMessages))]
	for i in range(len(possibleMessageScores)):
		inserted = False
		for j in range(len(idx)):
			if idx[j] == i:
				inserted = True
				break #in the right spot if less than everything above it
			if possibleMessageScores[i] > possibleMessageScores[idx[j]]:
				idx.remove(i)
				idx.insert(j, i)
				inserted = True
				break
		if not inserted:
			idx.remove(i)
			idx.append(i)

	possibleMessages = np.array(possibleMessages)[idx]
	possibleMessageScores = np.array(possibleMessageScores)[idx]

	#remove duplicate messages

	#filter scores less than the cutoff
	for i in range(len(possibleMessageScores) - 1, -1, -1):
		if len(possibleMessageScores) <= minOutput: #don't remove values if it would put us below the minimum output
			break
		if possibleMessageScores[i] < cutoff:
			possibleMessageScores = np.delete(possibleMessageScores, i)
			possibleMessages = np.delete(possibleMessages, i)
	
	#remove scores if there are more possible messages than the maximum allowed ouput. lower scores go first
	for i in range(len(possibleMessageScores) - 1, -1, -1):
		if len(possibleMessageScores) <= maxOutput or maxOutput == -1: #don't remove values if we are below or at the maximum output
			break
		possibleMessageScores = np.delete(possibleMessageScores, i)
		possibleMessages = np.delete(possibleMessages, i)

	return possibleMessages, possibleMessageScores

def writePredictionFile(originalMessage, possibleMessages, possibleMessageScores, targetFilePath = "./textPrediction.txt"):
	'''
	Writes predictions to a readable text file

	Parameters:
		originalMessage (str): predicted decryption from a trained decryption model
		possibleMessages (list): list of all possible messages given the predicted message and decryption model analysis
		possibleMessageScores (list): scores for each of the possible messages
		targetFilePath (str): path of the file to write to / create
	'''
	print(f"Writing Predictions to {targetFilePath}")
	if not len(possibleMessages) == len(possibleMessageScores):
		raise Warning("Length of possible messages is not the same as length of possible scores")
	textToWrite = ""
	textToWrite += f"Breaker Output:\n\t{originalMessage}\n\n"
	textToWrite += f"Possible Messages:\n"
	for i in range(len(possibleMessages)):
		textToWrite += f"\t{possibleMessages[i]}\n\tConfidence: {possibleMessageScores[i]}\n"
	
	with open(targetFilePath, "w") as targetFile:
		targetFile.write(textToWrite)



def predictUserInput(charClassifier, trainingType, includePredictedTextAsWords = False):
	'''
	Allows the user to input messages to encrypt, decryptes them using the provided decryption model, predicts possible messages based on its output an analysis, and then writes the results to a text file.

	Parameters:
		charClassifier (clf): trained decryption model
		trainingType (str): data format that the decryption model was trained on
		includePredictedTextAsWords (bool): should the predicted words be considered valid words when predicting possible messages
	'''
	#get message to encrypt
	messageToEncrypt = ""
	while messageToEncrypt == "":
		messageToEncrypt = input("Enter a message to encrypt:\n\t")
		
	#get probability cutoff value
	cutoff = -1
	while cutoff < 0 or cutoff > 1:
		try:
			cutoff = float(input("Probability cutoff: "))
		except:
			print("Invalid Input. Enter a number between 0.0 and 1.0")
		if cutoff < 0 or cutoff > 1:
			print("Invalid Input. Enter a number between 0.0 and 1.0")
		
	#get min output
	minOutput = -1
	while minOutput < 0:
		try:
			minOutput = int(input("Minimum Outputs: "))
		except:
			print("Invalid Input. Enter a positive integer.")
		if minOutput < 0:
			print("Invalid Input. Enter a positive integer.")

	#get max output
	maxOutput = 0
	while maxOutput == 0:
		try:
			maxOutput = int(input("Maximum Outputs (value of -1 = infinity): "))
		except:
			print("Invalid Input. Enter a positive integer or -1.")
		if maxOutput <= 0 and not maxOutput == -1:
			print("Invalid Input. Enter a positive integer or -1.")

	result = cipherBreaker.encryptAndBreak(charClassifier, messageToEncrypt, trainingType)
	print("Decrypted Message:\n\t" + result + "\n")

	pm, pms = predict(clf, result, trainingType, cutoff, minOutput, maxOutput, includePredictedTextAsWords)
	writePredictionFile(result, pm, pms)


trainingType = "uncompressed"
print("Loading Model...")
#clf = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
clf = charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-3.pkl")

predictUserInput(clf, trainingType, includePredictedTextAsWords = True)