from nltk.util import ngrams
from tokenizer import word_tokenize
from collections import defaultdict
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
        self.tags_prob = defaultdict(float) #Dictionary to store conditional probability of tag trigrams
        self.words_tags_prob = defaultdict(float) #Dictionary to store conditional probability of word-tag pairs
    
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
            self.tags_prob[(i,j,k)]=float(self.trigrams_dict[(i,j,k)])/float(self.bigrams_dict[(i,j)])
    
    def calc_R(self): #Function to calculate word-tag probabilities
        for word,tag in self.words_tags:
            self.words_tags_prob[(word,tag)]=float(self.words_tags_dict[(word,tag)])/float(self.unigrams_dict[(tag,)])

    def test(self): #Test function

        sent = "START START The grand jury commented. STOP"
        tag = "START START AT JJ NN VBD . STOP"
        prob_tag = 1.0
        prob_words = 1.0
        tot_prob = 1.0
        sent_tokens = [j for i in word_tokenize(sent) for j in i]
        tag_tokens = [j for i in word_tokenize(tag) for j in i]
        for i in range(2,len(tag_tokens)-1):
            prob_tag*=self.tags_prob[(tag_tokens[i-2],tag_tokens[i-1],tag_tokens[i])]
        for i,j in zip(sent_tokens,tag_tokens):
            prob_words*=self.words_tags_prob[(i,j)]
        tot_prob=prob_tag*prob_words
        print (tot_prob)
        
        print (self.viterbi())

    def viterbi(self):
        sent = "The grand jury commented." #Input sentence
        tokens=[j for i in word_tokenize(sent) for j in i] #Tokens of input sentence
        n = len(tokens) #Length of input sentence
        phi = [defaultdict(float) for i in range(n+1)] #List of dictionaries to store max probability of preceding tag sequence at a given postion
        back_ptr = [defaultdict(str) for i in range(n+1)] #List of dictionaries to store backpointers (tags) that maximises phi
        phi[0][('START','START')] = 1.0 #Initialise phi for START tags
        
        #Iterate through all tokens all store max probabilities and arguments
        for k in range(1,n+1): 
            x=tokens[k-1] 
            #Sample space 'S' for each tag w,u,v  
            U=self.tags_set
            V=self.tags_set
            W=self.tags_set
            if k==1: #Both preceding tags for first word are START in a trigram model
                W=('START')
                U=('START')
            if k==2: #One of the prceding tags for second word is START in a trigram model
                W=('START')
            for u in U:
                for v in V:
                    max_prob = 0.0 #Stores maximum probability for each word
                    max_arg = '' #Stores tag that maximises probability 
                    for w in W:
                        temp_prob=phi[k-1][(w,u)]*self.tags_prob[(w,u,v)]*self.words_tags_prob[(x,v)] #Calculate probability for tag sequence at position k
                        if temp_prob>max_prob: #Store maximum probability and tag 
                            max_prob=temp_prob
                            max_arg=w
                    phi[k][(u,v)]=max_prob #Store maximum probability for positon k and tags u,v
                    back_ptr[k][(u,v)]=max_arg #Store tag that maximises probability (backpointer)

        #Back Track to determine tag sequence
        max_prob = 0.0
        max_u = ''
        max_v = ''
        for u in self.tags: 
            for v in self.tags:
                temp_prob = phi[n][(u,v)]*self.tags_prob[(u,v,'STOP')]
                if prob>max_pi:
                    max_pi=temp_prob
                    max_u=u
                    max_v=v

        t = [None]*(n+1) #Initialise tag sequence
        t[n-1] = max_u
        t[n] = max_v
        for k in range(n-2,0,-1):
            t[k]=back_ptr[k+2][(t[k+1],t[k+2])]
        return t
   
if __name__=="__main__":
    pos_tag = pos_tagger()
    pos_tag.tokenize()
    pos_tag.init_tags()
    pos_tag.init_words_tags()
    pos_tag.init_dict()
    pos_tag.calc_Q()
    pos_tag.calc_R()
    pos_tag.test()
