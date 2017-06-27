from nltk.probability import ConditionalFreqDist as CFD
from tokenizer import word_tokenize
import re

class pos_tagger:
    def __init__(self):
        self.sents = [] #List containing tokenized sentences
        self.tags = [] #List containing all tags used in brown_tagged.txt
        self.tokens_tags = [] #List containing token and respective tag as a tuple
        self.cfd = None #Conditional Frequency Distribution
    def tokenize(self):
        text = open("brown.txt","r").read()
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
    def initialise_tokens_tags(self):
        tagged_sents = open("brown_tagged.txt").read().splitlines()
        tokens_tags = []
        for tagged_sent in tagged_sents:
            tagged_words = tagged_sent.split()
            for tagged_word in tagged_words:
                m = re.search('(.*)_(.*)',tagged_word)
                tokens_tags.append((m.group(1),m.group(2)))
            self.tokens_tags.append(tokens_tags)
            tokens_tags = []
    def initialise_cfd(self):
        self.cfd = CFD(j for i in self.tokens_tags for j in i)
            
pos_tag = pos_tagger()
pos_tag.tokenize()
pos_tag.initialise_tags()
pos_tag.initialise_tokens_tags()
pos_tag.initialise_cfd()
