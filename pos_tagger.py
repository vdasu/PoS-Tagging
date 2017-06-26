from tokenizer import word_tokenize
import re

class viterbi:
    def __init__(self):
        self.sents = []
        self.tags = []
    def tokenize(self,text):
        self.sents = word_tokenize(text)
        for sent in self.sents:
            sent[:0] = ['START','START'] #Append START token to beginning of list
            sent.append('STOP') #Append STOP token to end of list
    def initialise_tags(self):
        sents = open("brown_tagged.txt").read().splitlines()
        for sent in sents:
            words = sent.split()
            for word in words:
                m = re.search('_(.*)',word)
                self.tags.append(m.group(1))
        self.tags = set(self.tags)
    def display(self):
        for i in self.tags:
            print (i)
            


pos_tag = viterbi()
text = open("brown.txt","r").read()
pos_tag.tokenize(text)
pos_tag.initialise_tags()
pos_tag.display()
