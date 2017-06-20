import re

#Funtion takes raw text as input and returns tokens
def word_tokenize(text): 
    tokens = [];
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s',text) #Regex to tokenize sentences
    for txt in sents:
        tokens.append(re.findall(r"[-]{2}|\w+(?:\.(?=\S)|-\w+)+|[\w']+|[^\w\s]",txt)) #Regex to tokenize words
    return tokens #List of tokens of each sentence in raw text returned
