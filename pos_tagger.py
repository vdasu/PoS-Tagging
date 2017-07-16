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
        sents = open("brown_trigram.txt").read().splitlines()[:55000]
        for sent in sents:
            words = sent.split()
            for word in words:
                m = re.search('_(.*)',word)
                self.tags.append(m.group(1))
        self.tags_set = set(self.tags) #List of all unique tags in corpus
    
    def init_words_tags(self): #Function to initialise word-tag pairs
        tagged_sents = open("brown_trigram.txt").read().splitlines()[:55000]
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
        		W = ('START',)
        		U = ('START',)
        	if k==2: #One of the preceding tags for second word is START in a trigram model
        		W = ('START',)
        	for u in U: #Tag at position k-1
        		for v in V: #Tag at position k
        			max_prob = 0.0 #Stores maximum probability for each word
        			max_arg = "" #Stores tag that maximises probability 
        			for w in W: #Tag at position k-2
        				if (self.R[(x,v)]) == 0.0:
        					continue
        				if psi[k-1][(w,u)]!=0.0:
        					temp_prob = psi[k-1][(w,u)]*self.Q[(w,u,v)]*self.R[(x,v)] #Calculate probability for tag sequence at position k
        				if temp_prob>max_prob:
        					max_prob = temp_prob
        					max_arg = w     
        			if max_prob!=0.0:
        				psi[k][(u,v)] = max_prob #Store maximum probability for positon k and tags u,v
        				back_ptr[k][(u,v)] = max_arg #Store tag that maximises psi (backpointer)
        
        #Back Track to determine tag sequence
        max_prob = 0.0 
        max_u = "" 
        max_v = ""
        for u in self.tags_set: #Tag at position n-1
        	for v in self.tags_set: #Tag at position n
        		if self.Q[(u,v,'STOP')]*psi[n][(u,v)]!=0.0:
        			temp_prob = psi[n][(u,v)]*self.Q[(u,v,'STOP')]
        			if temp_prob>max_prob:
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

        print ("Number of words in Brown corpus: 1333212")
        print ("Number of unique tags in Brown corpus: 474")

        f = open("input.txt","r").read()
        
        file_info = stat("input.txt")

        print ("Size of test file: ",file_info.st_size)

        sents_tokens = word_tokenize(f)
        print ("Number of tags to be tokenized: ",len([j for i in sents_tokens for j in i]))

        t0 = time()
        tagger = TrigramTagger(brown.tagged_sents()[:55000])
        t1 = time()
        nltk_train_time = t1-t0
        print ("Time taken by NLTK for training: ",nltk_train_time)

        nltk_tags = []
        t0 = time()
        for sent in sents_tokens:
            nltk_tags.append(tagger.tag(sent))
        t1 = time()
        nltk_tag_time = t1-t0
        print ("Time taken by NLTK to tag text: ",nltk_tag_time)

        t0 = time()
        self.tokenize()
        self.init_tags()
        self.init_words_tags()
        self.init_dict()
        self.calc_Q()
        self.calc_R()
        t1 = time()
        pos_train_time = t1-t0

        print ("Time taken by pos_tagger to train: ",pos_train_time)

        pos_tagger_tags = []
        t0 = time()
        for sent in sents_tokens:
            pos_tagger_tags.append(self.viterbi(sent))
        t1 = time()
        pos_tag_time = t1-t0
        print ("Time taken by pos_tagger to tag: ",pos_tag_time)

        if nltk_train_time <pos_train_time:
            print ("Training time of NLTK is less than pos_tagger by: ",abs(nltk_train_time-pos_train_time))
        else:
            print ("Training time of pos_tagger is less than NLTK by: ",abs(nltk_train_time-pos_train_time))

        if nltk_tag_time<pos_tag_time:
            print ("Tagging time of NLTK is less than pos_tagger by: ",abs(nltk_tag_time-pos_tag_time))
        else:
            print ("Tagging time of pos_tagger is less than NLTK by: ",abs(nltk_tag_time-pos_tag_time))

        nltk_tag_count = defaultdict(int)
        for i in nltk_tags:
            for j in i:
                nltk_tag_count[j[1]]+=1

        pos_tag_count = defaultdict(int)
        for i in pos_tagger_tags:
            for j in i:
                pos_tag_count[j[1]]+=1

        print ("POS tags generated by NLTK: ")
        for i in nltk_tag_count.items():
            print (i)

        print ("POS tags generated by pos_tagger: ")
        for i in pos_tag_count.items():
            print (i)

        print ("Number of unique tags generated by NLTK: ",len([i for i in nltk_tag_count.keys()]))

        print ("Number of unique tags generated by pos_tagger: ",len([i for i in pos_tag_count.keys()]))

        print ("NLTK failed to tag",nltk_tag_count[None],"tokens")

        print ("pos_tagger failed to tag",pos_tag_count[''],"tokens")

        if nltk_tag_count[None]>pos_tag_count['']:
        	print ("pos_tagger tagged",abs(nltk_tag_count[None]-pos_tag_count['']),"more tokens than NLTK")
        else:
        	print ("NLTK tagged",abs(nltk_tag_count[None]-pos_tag_count['']),"more tokens than pos_tagger")

        tagged_sents = open("input_tagged.txt","r").read().splitlines()
        tags = []
        for sent in tagged_sents:
        	words = sent.split()
        	for word in words:
        		m = re.search('(.*)_(.*)',word)
        		tags.append(m.group(2))
        
        n_tags = [j[1] for i in nltk_tags for j in i]
        nltk_count = 0
        for x,y in zip(n_tags,tags):
        	if x==y:
        		nltk_count+=1

        len_tokens = len([j for i in sents_tokens for j in i])

        print ("NLTK accurately tagged",nltk_count,"tokens")
        print ("NLTK accuracy score: ",float(nltk_count)/float(len_tokens))

        p_tags = [j[1] for i in pos_tagger_tags for j in i]
        pos_count = 0
        for x,y in zip(p_tags,tags):
        	if x==y:
        		pos_count+=1

        print ("pos_tagger accurately tagged",pos_count,"tokens")
        print ("pos_tagger accuracy score: ",float(pos_count)/float(len_tokens))

        if nltk_count>pos_count:
        	print ("NLTK accurately tagged",abs(nltk_count-pos_count),"more tokens than pos_tagger")
        else:
        	print ("pos_tagger accurately tagged",abs(nltk_count-pos_count),"more tokens than NLTK")
       

if __name__=="__main__":
    pos_tag = pos_tagger()
    pos_tag.baseline_tagger()