from time import time
from tokenizer import word_tokenize
    
text = open("brown.txt","r").read()
t1 = time()
tokens = word_tokenize(text)
t2 = time()
print ("Time taken to tokenize: ",t2-t1)
input("Press Enter to view tokens")
print (tokens)
        
