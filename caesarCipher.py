import cipher

global abc
global ABC
global str_num
global sym

abc = cipher.abc
ABC = cipher.ABC
str_num = cipher.str_num
sym = cipher.sym[1:] #don't include space

def to_cipher(message, rshift = 1):
    finalMessage = ""
    combinedChars = abc + ABC + str_num + sym
    for char in message:
        if not char in combinedChars and not char == " ":
            raise Exception(f"Character {char} is not a supported character")
        elif char == " ":
            finalMessage += char
        else:
            finalMessage += combinedChars[(combinedChars.index(char) + rshift) % len(combinedChars)]
    return finalMessage

def from_cipher(encryptedMessage, rshift = 1):
    finalMessage = ""
    combinedChars = abc + ABC + str_num + sym
    for char in encryptedMessage:
        if not char in combinedChars and not char == " ":
            print(f"Character {char} is not translatable")
            finalMessage += "_"
        elif char == " ":
            finalMessage += char
        else:
            finalMessage += combinedChars[(combinedChars.index(char) - rshift) % len(combinedChars)]
    return finalMessage