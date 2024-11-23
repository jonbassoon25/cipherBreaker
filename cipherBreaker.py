import numpy as np
import joblib

import cipher
import dataFormatter

def formatAsNumbers(encryptedChars):
	encryptedChars = dataFormatter.formatAsNumbers(encryptedChars).ravel()
	return np.array([[int(num) for num in nums] for nums in encryptedChars])

def score(actual, result):
	correct = 0
	for i in range(len(result)):
		if i >= len(actual):
			break
		if actual[i] == result[i]:
			correct += 1
	return round(correct/len(actual), 3)
	

trainingType = "uncompressed"
charClassifier = joblib.load("./CCCs/cipherCharacterClassifier.pkl")
#charClassifier = joblib.load("./CCCs/saved/clf-2.pkl")

while True:
	messageToEncrypt = ""
	while messageToEncrypt == "":
		messageToEncrypt = input("Enter a message to encrypt:\n\t")
	encryptedMessage = cipher.to_cipher(messageToEncrypt)
	#print("\nDecrypting Message...")
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
	if trainingType == "uncompressed":
		encryptedChars = formatAsNumbers(encryptedChars)
		print("Encrypted Message:\n\t" + "".join([str(thing) for thing in encryptedChars.ravel()]) + "\n")
	elif trainingType == "compressed":
		encryptedChars = dataFormatter.formatAsCompressed(np.array([encryptedChars]))[0]
		print("Encrypted Message:\n\t" + encryptedMessage)

	result = "".join(charClassifier.predict(encryptedChars))
	print("Decrypted Message:\n\t" + result)
	print("Decryption Score: " + str(score(messageToEncrypt, result)))
	print("\n")