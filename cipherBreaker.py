import numpy as np
import joblib

import cipher
import dataFormatter

def formatAsNumbers(encryptedChars):
	'''
	Formats the given encrypted chars as numbers of length 24

	Parameters:
		encryptedChars (npArray): array of encrypted characters
	
	Returns:
		(npArray): array of encrypted characters with each character being a 24 digit number
	'''
	encryptedChars = dataFormatter.formatAsNumbers(encryptedChars).ravel()
	return np.array([[int(num) for num in nums] for nums in encryptedChars])

def formatAsCompressed(encryptedChars):
	'''
	Formats the given encrypted chars as arrays of ordinal numbers

	Parameters:
		encryptedChars (npArray): array of encrypted characters
	
	Returns:
		(npArray): array of encrypted characters with each character being an array of length 32 of ordinal numbers representing each character of the encrypted character
	'''
	return dataFormatter.formatAsCompressed(encryptedChars)

def score(actual, result):
	'''
	Scores the similarity of the result string against the actual string

	Parameters:
		actual (str): real string
		result (str): predicted string
	
	Returns:
		(float): similarity of the result string and the actual string to 3 decimal places
	'''
	correct = 0
	for i in range(len(result)):
		if i >= len(actual):
			break
		if actual[i] == result[i]:
			correct += 1
	return round(correct/len(actual), 3)

def breakCipher(clf, encryptedMessage, type):
	'''
	Uses the provided classifier model to decrypt the encrypted message

	Parameters:
		clf (clf): trained decryption model
		encryptedMessage (str): message to decrypt
		type (str): data format that the classifier was trained on
	
	Returns:
		(str): predicted decryption of message
	'''
	#get each character of the encrypted message on its own
	encryptedChars = []
	i = 0
	while i < len(encryptedMessage):
		if encryptedMessage[i] == " ":
			encryptedChars.append(encryptedMessage[i:i+3])
			i += 2
		else:
			encryptedChars.append(encryptedMessage[i])
		i += 1

	encryptedChars = np.array(["".join(encryptedChars[i:i+12]) for i in range(0, len(encryptedChars), 12)])
	
	#format data for ml, match to training data format
	if type == "uncompressed":
		encryptedChars = formatAsNumbers(encryptedChars)
		print("Encrypted Message:\n\t" + "".join([str(thing) for thing in encryptedChars.ravel()]) + "\n")
	elif type == "compressed":
		encryptedChars = formatAsCompressed(np.array([encryptedChars]))[0]
		print("Encrypted Message:\n\t" + encryptedMessage)
		
	result = "".join(clf.predict(encryptedChars))
	return result

def encryptAndBreak(clf, message, type):
	'''
	Encryptes the given message and decrypts it with the given classifier

	Parameters:
		clf (clf): trained decryption model
		message (str): message to encrypt
		type (str): data format the classifier was trained on
	
	Returns:
		(str): predicted decryption of the encryption of message
	'''
	encryptedMessage = cipher.to_cipher(message)
	return breakCipher(clf, encryptedMessage, type)

def decryptUserMessages(charClassifier, trainingType):
	'''
	Allows the user to input messages to encrypt and displays the encrypted message as well as the predicted decryption

	Parameters:
		charClassifier (clf): trained decryption model
		trainingType (str): data format that the model was trained on
	'''
	while True:
		messageToEncrypt = ""
		while messageToEncrypt == "":
			messageToEncrypt = input("Enter a message to encrypt:\n\t")
		
		result = encryptAndBreak(charClassifier, messageToEncrypt, trainingType)

		print("Decrypted Message:\n\t" + result)
		print("Decryption Score: " + str(score(messageToEncrypt, result)))
		print("\n")

'''
trainingType = "uncompressed"
#charClassifier = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
charClassifier = joblib.load(f"./CCCs/saved/{trainingType}/clf-1.pkl")
decryptUserMessages(charClassifier, trainingType)
'''