Created by Jonathan Hanson

This project aims to train a ML model capable of decrypting a cipher that was developed by me 2 years prior to the start of this project

Encryption Types
    Uncompressed - uncompressed cipher, removed any variability in the length by uncompressing the output into its number form that is always 24 characters long
        Each character is represented by an array of length 24 with each element in the array being a single digit from the encrypted message
    Compressed - compressed cipher, final compressed cipher that has a length between 12 and 36. 
        Each character is represented by an array of length 36 with each element in the array being the ordinal number of a character of the encrypted message
