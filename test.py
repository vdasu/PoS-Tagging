from nltk.corpus import brown
from tokenizer import word_tokenize
    
for i,j in zip(brown.sents(),brown.tagged_sents()):
    tokens = word_tokenize(' '.join(i))
    for x,y in zip(i,(str(z[0]) for z in j)):
        if x!=y:
            print ("MISMATCH")
            exit()
        else:
            print (x," ",y)
        
