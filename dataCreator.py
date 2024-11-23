import numpy as np
import cipher
import util

def createSingleCharData(char, quantity, printStep = -1):
	'''
		returns a numpy array of encryptions of the given char with a shape of (quantity,)

		Parameters:
			char (str or char): character to encrypt
			quantity (int): number of times to encrypt the char
			printStep (int): number of encryptions between encryption progress updates

		Returns:
			npArray: quantity encryptions of char
	'''
	charData = np.array([" " * int(24/2 * 3) for i in range(quantity)])

	if printStep == -1:
		for i in range(quantity):
			charData[i] = cipher.to_cipher(char)
	else:
		index = 0
		for i in range(quantity):
			charData[i] = cipher.to_cipher(char)
			index += 1
			if index % printStep == 0:
				print(f"Finished {index} Encryptions")
	return charData

def addType(char):
	'''
		This function takes 1 character as input and returns it with a type identifier

		Parameters:
			char (str): input character

		Returns:
			str: char-type
	'''
	if cipher.abc.count(char) > 0:
		return f"{char}-lor" #lower
	if cipher.ABC.count(char) > 0:
		return f"{char}-upr" #upper
	if cipher.str_num.count(char) > 0:
		return f"{char}-num" #number
	if cipher.sym.count(char) > 0:
		return f"{char}-sym" #symbol
	else:
		return f"{char}-und" #undefined

def createData(quantity = 10_000, dtype = "all", verbose = False, printStep = -1):
	'''
		Creates and saves encrypted letter data to ./arrays/encp

		Parameters:
			quantity (int): quantity of encryptions for each letter
			dtype (str): type of data to encrypt (lowercase, uppercase, number, symbol, or all)
			verbose (bool): should progress updates be printed to the console when each char starts encryption
			printStep (int): number of encryptions of each letter between encryption progress updates
	'''
	cipherChars = []
	if dtype == "lowercase":
		cipherChars = cipher.abc
	elif dtype == "uppercase":
		cipherChars = cipher.ABC
	elif dtype == "number":
		cipherChars = cipher.str_num
	elif dtype == "symbol":
		cipherChars = cipher.sym
	elif dtype == "all":
		twoDCipherChars = [cipher.abc, cipher.ABC, cipher.str_num, cipher.sym]
		for array in twoDCipherChars:
			for char in array:
				cipherChars.append(char)
		
	for char in cipherChars:
		if verbose:
			print(f"Encrypting {quantity} of {char}")
		util.saveArray(f"encp/{addType(char)}", createSingleCharData(char, quantity, printStep))