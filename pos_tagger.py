from nltk.util import ngrams
from tokenizer import word_tokenize
from collections import defaultdict
from time import time
from os import stat
import re

class pos_tagger:
    
    def __init__(self):
        self.sents = [] #List containing tokenized sentences
        self.tags = [] #List containing all tags used in Brown Corpus
        self.tags_set = [] #List containing set of all tags used in Brown Corpus
        self.words_tags = [] #List containing token and respective tag as a tuple
        self.tagged_sents = [] #List containing tokenized tagged sentences
        self.trigrams_as_bigrams = [] #List containing trigrams of tokens as bigrams 
        self.words_tags_dict = defaultdict(int) #Dictionary to store count of word-tag pairs
        self.trigrams_dict = defaultdict(int) #Dictionary to store count of tags as trigrams
        self.bigrams_dict = defaultdict(int) #Dictionary to store count of tags as bigrams
        self.unigrams_dict = defaultdict(int) #Dictionary to store count of tags as unigrams
        self.Q = defaultdict(float) #Dictionary to store conditional probability of tag trigrams
        self.R = defaultdict(float) #Dictionary to store conditional probability of word-tag pairs
    
    def tokenize(self): #Function to tokenize text
        text = open("brown.txt","r").read()
        self.sents = word_tokenize(text)
        for sent in self.sents:
            sent[:0] = ['START','START'] #Append START token to beginning of list
            sent.append('STOP') #Append STOP token to end of list
    
    def init_tags(self): #Function to initialise tags
        sents = open("brown_trigram.txt").read().splitlines()
        for sent in sents:
            words = sent.split()
            for word in words:
                m = re.search('_(.*)',word)
                self.tags.append(m.group(1))
        self.tags_set = set(self.tags) #List of all unique tags in corpus
    
    def init_words_tags(self): #Function to initialise word-tag pairs
        tagged_sents = open("brown_trigram.txt").read().splitlines()
        words_tags = []
        for tagged_sent in tagged_sents:
            tagged_words = tagged_sent.split()
            for tagged_word in tagged_words:
                m = re.search('(.*)_(.*)',tagged_word)
                words_tags.append((m.group(1),m.group(2)))
            self.words_tags.append(words_tags)
            words_tags = []
        self.tagged_sents = self.words_tags
        self.words_tags = [j for i in self.words_tags for j in i]
    
    def init_dict(self): #Function to initialised dictionaries
        for i in self.words_tags:
            self.words_tags_dict[i]+=1   
        trigrams_tags = ngrams(self.tags,3)
        for i in trigrams_tags:
            self.trigrams_dict[i]+=1
        bigrams_tags = ngrams(self.tags,2)
        for i in bigrams_tags:
            self.bigrams_dict[i]+=1
        unigrams_tags = ngrams(self.tags,1)
        for i in unigrams_tags:
            self.unigrams_dict[i]+=1
    
    def calc_Q(self): #Function to calculate tag probabilities
        trigrams_tags = ngrams(self.tags,3)
        for i,j,k in trigrams_tags:
            self.Q[(i,j,k)]=float(self.trigrams_dict[(i,j,k)])/float(self.bigrams_dict[(i,j)])
    
    def calc_R(self): #Function to calculate word-tag probabilities
        for word,tag in self.words_tags:
            self.R[(word,tag)]=float(self.words_tags_dict[(word,tag)])/float(self.unigrams_dict[(tag,)])

    def test(self): #Test function

        sent = "START START The grand jury commented. STOP"
        tag = "START START AT JJ NN VBD  . STOP" 
        prob_tag = 1.0
        prob_words = 1.0
        tot_prob = 1.0
        sent_tokens = [j for i in word_tokenize(sent) for j in i]
        tag_tokens = [j for i in word_tokenize(tag) for j in i]
        for i in range(2,len(tag_tokens)-1):
            prob_tag*=self.Q[(tag_tokens[i-2],tag_tokens[i-1],tag_tokens[i])]
        for i,j in zip(sent_tokens,tag_tokens):
            prob_words*=self.R[(i,j)]
        tot_prob=prob_tag*prob_words
        print (tot_prob)
        

    def viterbi(self,tokens): #Function returns list containing tokens and their respective POS tags
        
        n = len(tokens) #Length of input sentence
        psi = [defaultdict(float) for i in range(n+1)] #List of dictionaries to store max probability of preceding tag sequence at a given postion
        back_ptr = [defaultdict(str) for i in range(n+1)] #List of dictionaries to store backpointers (tags) that maximises psi
        psi[0][('START','START')] = 1.0 #Base case

        #Iterate through all tokens all store max probabilities and max arguments
        for k in range(1,n+1):
            x = tokens[k-1]
            #Sample space 'S' for each tag w,u,v  
            W = self.tags_set
            U = self.tags_set
            V = self.tags_set
            if k==1: #Both preceding tags for first word are START in a trigram model
                W = ('START')
                U = ('START')
            if k==2: #One of the preceding tags for second word is START in a trigram model
                W = ('START')
            for u in U: #Tag at position k-1
                for v in V: #Tag at position k
                    max_prob = 0.0 #Stores maximum probability for each word
                    max_arg = "" #Stores tag that maximises probability 
                    for w in W: #Tag at position k-2
                        temp_prob = psi[k-1][(w,u)]*self.Q[(w,u,v)]*self.R[(x,v)] #Calculate probability for tag sequence at position k
                        if temp_prob>max_prob:
                            if temp_prob!=0.0:
                                max_prob = temp_prob
                                max_arg = w 
                    psi[k][(u,v)] = max_prob #Store maximum probability for positon k and tags u,v
                    back_ptr[k][(u,v)] = max_arg #Store tag that maximises psi (backpointer)

            #Back Track to determine tag sequence
            max_prob = 0.0 
            max_u = "" 
            max_v = ""
            for u in self.tags_set: #Tag at position n-1
                for v in self.tags_set: #Tag at position n
                    temp_prob = psi[n][(u,v)]*self.Q[(u,v,'STOP')]
                    if temp_prob>max_prob:
                        if temp_prob!=0.0:
                            max_prob = temp_prob
                            max_u = u
                            max_v = v

            t = [None]*(n+1) #Initialise tag sequence
            t[n] = max_v
            t[n-1] = max_u
            for k in range(n-2,0,-1):
                t[k] = back_ptr[k+2][(t[k+1],t[k+2])] #Back track to complete tag sequence
            
            tagged_sent = [] #List to store token-tag tuple 
            for token,tag in zip(tokens,t[1:]):
                tagged_sent.append((token,tag))
            return tagged_sent


    def baseline_tagger(self):

        from nltk.corpus import brown
        from nltk.tag import TrigramTagger 

        f = open("input.txt","r").read()
        
        file_info = stat("input.txt")

        print ("Size of test file: ",file_info.st_size)

        t0 = time()
        tagger = TrigramTagger(brown.tagged_sents())
        t1 = time()
        print ("Time taken by NLTK for training: ",t1-t0)

        sents_tokens = word_tokenize(f)
        nltk_tags = []
        t0 = time()
        for sent in sents_tokens:
            nltk_tags.append(tagger.tag(sent))
        t1 = time()
        print ("Time taken by NLTK to tag text: ",t1-t0)

        pos_tagger_tags = []
        for sent in sents_tokens:
            pos_tagger_tags.append(self.viterbi(sent))


   
if __name__=="__main__":
    pos_tag = pos_tagger()
    pos_tag.tokenize()
    pos_tag.init_tags()
    pos_tag.init_words_tags()
    pos_tag.init_dict()
    pos_tag.calc_Q()
    pos_tag.calc_R()
    pos_tag.test()
    pos_tag.baseline_tagger()