import numpy as np
import cipher
import util

def formatAsNumbers(encryptedDataArray):
	'''
	Uncompresses each encrypted value in the given encrypted data array

	Parameters:
		encryptedDataArray (npArray): 2d array of encrypted characters
	
	Returns:
		npArray: encryptedDataArray with each encrypted value in its uncompressed form
	'''
	initialShape = encryptedDataArray.shape
	encryptedDataArray = encryptedDataArray.ravel()
	finalDataArray = np.array([" " * 24 for i in range(encryptedDataArray.shape[0])])
	for i in range(finalDataArray.shape[0]):
		finalDataArray[i] = cipher.uncompress(encryptedDataArray[i])
	return finalDataArray.reshape(initialShape)

def formatAsCompressed(encryptedDataArray):
	'''
	Formats each encrypted value in encryptedDataArray as an array of length 36 with each array element being a character of the encrypted value

	Parameters:
		encryptedDataArray (npArray): 2d array of encrypted characters

	Returns:
		npArray: encryptedDataArray with each value being an array of length 36 containing the ordinal numbers of each characeter of the encrypted value
	'''
	initialShape = encryptedDataArray.shape
	encryptedDataArray = encryptedDataArray.ravel()
	finalArray = np.array([np.zeros(36, dtype="uint8") for i in range(len(encryptedDataArray))])
	for i in range(len(encryptedDataArray)): #found max length of 26 per translation through translation testing, calculated max is 36
		uncompressed = [ord(char) for char in encryptedDataArray[i]]
		for j in range(len(finalArray[i])):
			if j < len(uncompressed):
				finalArray[i][j] = uncompressed[j]
	return finalArray.reshape((initialShape[0], initialShape[1], 36))

def createNumberData():
	'''Creates uncompressed encryptions from ./arrays/encp and saves them to ./arrays/num_encp'''
	encp_abc, encp_ABC, encp_num, encp_sym = util.loadEncpData()
	util.saveArray("num_encp/abc-lor", formatAsNumbers(encp_abc))
	util.saveArray("num_encp/ABC-upr", formatAsNumbers(encp_ABC))
	util.saveArray("num_encp/num-num", formatAsNumbers(encp_num))
	util.saveArray("num_encp/sym-sym", formatAsNumbers(encp_sym))

def createCompressedData():
	'''Creates compressed encryptions from ./arrays/encp and saves them to ./arrays/com_encp'''
	encp_abc, encp_ABC, encp_num, encp_sym = util.loadEncpData()
	util.saveArray("com_encp/abc-lor", formatAsCompressed(encp_abc))
	util.saveArray("com_encp/ABC-upr", formatAsCompressed(encp_ABC))
	util.saveArray("com_encp/num-num", formatAsCompressed(encp_num))
	util.saveArray("com_encp/sym-sym", formatAsCompressed(encp_sym))