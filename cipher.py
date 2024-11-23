import time
import random
cipher_num = "00"

#declare global values
global abc
global ABC
global str_num
global sym
global abc_convert
global ABC_convert
global num_convert
global sym_convert

#set global values

#set all supported letters, numbers, and symbols
abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
ABC = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
str_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
sym = [" ", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "{", "}", "[", "]", "\\", "|", ";", ":", "'", "\"", ",", ".", "<", ">", "/", "?", "`", "~"]

#set conversion numbers
abc_convert = [80172599, 19018775, 55223788, 82169340, 79652671, 82581483, 84294070, 70323581, 68217152, 39173726, 41574762, 54594244, 63173486, 57087313, 83762575, 33749897, 69284039, 11780575, 42644065, 11343149, 82295201, 72871967, 29858213, 65717980, 46314573, 65540694]
ABC_convert = [57460057, 51834534, 63739554, 95099535, 74789106, 66823612, 92758169, 27580856, 64499828, 96959279, 14772520, 75320592, 82573996, 80172507, 35185382, 89382482, 70887567, 66194744, 96161381, 68765556, 38789629, 89881155, 70344414, 28098141, 90613848, 69915053]
num_convert = [45754997, 59461008, 51356057, 82877230, 94865623, 21367471, 67364218, 12628429, 19460772, 56610826]
sym_convert = [34126067, 58998412, 32875130, 79083743, 67862083, 44480042, 21831723, 28085888, 32167575, 14441370, 85498183, 15771041, 38874783, 93557146, 53741500, 14067993, 54366332, 97117593, 43797144, 91726098, 20635680, 58013905, 65799134, 95233936, 83236379, 13584113, 73063268, 75282796, 91660433, 67919505, 16491396, 28717900, 10824176]
undefined_convert = 11111111

def scramble(num):
	'''scrambles num, num can be string or int, num is returned as string'''
	num = str(num)
	num1 = []
	num2 = []
	for i in range(len(num)):
		if i < (len(num)/2):
			num1.append(num[i])
		else:
			num2.append(num[i])
	num1.reverse()
	num2.reverse()
	return ''.join(num1) + ''.join(num2)

def make_list(string):
	end_list = []
	for i in range(len(string)):
		end_list.append(string[i])
	return end_list

def special_list(string, length):
	#print(string)
	end_list = []
	for i in range(round(len(string)/length)):
		thing_to_add = ''
		for k in range(length):
			thing_to_add += string[k + (i * length)]
		end_list.append(thing_to_add)
	return end_list

def collide(short_num, long_num):
	'''
		collides num with another num
		nums can be strings or ints
		one num is returned as string
		long num should be 2 times the length of short num
	'''
	short_num = str(short_num)
	long_num = str(long_num)
	short_num = make_list(short_num)
	long_num = make_list(long_num)
	final_num = []
	for i in range(len(short_num)):
		final_num.append(short_num[i])
		final_num.append(long_num.pop(0))
		final_num.append(long_num.pop(0))
	return ''.join(final_num)

def encrypt(number):
	'''
		encrypts number by multiplying by a random one
		output is short num followed by long num
		number can be int or string
		output is list with 2 strings inside
	'''
	number = int(number)
	rand_num = random.randint(10000000, 99999999)
	number = scramble(number)
	end_number = str(int(rand_num) * int(number))
	while len(end_number) < 16:
		end_number = "0" + end_number

	return [str(scramble(rand_num)), end_number]

def convert_to_num(character):
	for i in range(len(sym)):
		if i < len(str_num):
			if str_num[i] == character:
				return num_convert[i]

		if i < len(abc):
			if abc[i] == character:
				return abc_convert[i]

			if ABC[i] == character:
				return ABC_convert[i]

		if i < len(sym):
			if sym[i] == character:
				return sym_convert[i]
				
	#print(f"Warning: Character {character} Not Found.")
	return -1

def compress(num):
	'''
		output is string
		input is string or int
	'''
	#print(num)
	num = str(num)
	num = make_list(num)
	result = ""
	for i in range(round(len(num)/2)):
		bit = num[i * 2] + num[1 + (i * 2)]
		#print(int(bit) + 33)
		if bit[0] == 0:
			del bit[0]
		#print(bit)
		if bit == "93" or bit == "94" or bit == "95" or bit == "96" or bit == "97" or bit == "98" or bit == "99":
			result += " " + bit
		else:
			result += chr(int(bit) + 33)
		
	return result
	
def uncompress(character):
	'''
		output is string
	'''
	result = ''
	ignore_step = 0
	for i in range(len(character)):
		#print(ord(character[i]))
		if character[i] == " ":
			result += (character[i + 1] + character[i + 2])
			ignore_step = 2
		elif ignore_step == 0:
			num = ord(character[i]) - 33
			num = str(num)
			if len(num) < 2:
				num = "0" + num
			result += num
		else:
			ignore_step -= 1
	return result

def convert_to_char(num):
	if num == undefined_convert:
		return "|und|"
	
	for i in range(len(sym)):
		if i < len(str_num):
			if num_convert[i] == num:
				return str_num[i]

		if i < len(abc):
			if abc_convert[i] == num:
				return abc[i]

			if ABC_convert[i] == num:
				return ABC[i]

		if i < len(sym):
			if sym_convert[i] == num:
				return sym[i]
				
	print(f"Code {num} Not Found")
	return '-'

def decrypt(key, letter):
	key = unscramble(key)
	return unscramble(round(int(letter) / int(key)))

def uncollide(num):
	'''
		num should be a string
		output is list with short num then long num in string format
	'''
	short_num = ""
	long_num = ""
	num = str(num)
	for i in range(round(len(num) / 3)):
		short_num += num[i * 3]
		long_num += num[1 + (i * 3)]
		long_num += num[2 + (i * 3)]
	return [short_num, long_num]

def unscramble(num):
	'''unscrambles num, num can be string or int, num is returned as int'''
	num = str(num)
	num1 = []
	num2 = []
	for i in range(len(num)):
		if i < (len(num)/2):
			num1.append(num[i])
		else:
			num2.append(num[i])
	num1.reverse()
	num2.reverse()
	return int(''.join(num1) + ''.join(num2))
	
def to_cipher(message, verbose = False):
	messageLength = len(message)
	message = make_list(message)
	pastTime = time.time()
	for i in range(len(message)):
		num = convert_to_num(message.pop(0))
		if num != -1:
			num = encrypt(num)
		else:
			continue
		message.append(compress(collide(num[0], num[1])))
		if verbose and i % 5000 == 0:
			print(f"Progress: {(100 * i) // messageLength}%\t{i}/{messageLength}\tEstimated Time Remaining: {((messageLength - i) * ((time.time() - pastTime)))//5000}s")
			pastTime = time.time()
	return ''.join(message)

def from_cipher(cipher):
	cipher = uncompress(cipher)
	cipher = special_list(cipher, 24)
	for i in range(len(cipher)):
		num = uncollide(cipher.pop(0))
		cipher.append(convert_to_char(decrypt(num[0], num[1])))
	return ''.join(cipher)