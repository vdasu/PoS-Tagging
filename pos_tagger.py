from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import ConditionalProbDist as CPD
from nltk.probability import MLEProbDist
from nltk.util import ngrams
from tokenizer import word_tokenize
import re

class pos_tagger:
    def __init__(self):
        self.sents = [] #List containing tokenized sentences
        self.tags = [] #List containing all tags used in Brown Corpus
        self.tags_set = [] #List containing set of all tags used in Brown Corpus
        self.tags_tokens = [] #List containing token and respective tag as a tuple
        self.cfddist_tagswords = None #Conditional Frequency Distribution of words over tags
        self.cpddist_tagswords = None #Conditional Probability Distribution of words over tags
        self.cfddist_tags = None #Conditional Frequency Distribution of tags as trigrams
        self.cpddist_tags = None #Conditional Probability Distribution of tags as trigrams
        self.trigrams_as_bigrams = [] #List containing trigrams of tokens as bigrams 
    def tokenize(self):
        text = open("brown.txt","r").read()
        self.sents = word_tokenize(text)
        for sent in self.sents:
            sent[:0] = ['START','START'] #Append START token to beginning of list
            sent.append('STOP') #Append STOP token to end of list
    def init_tags(self):
        sents = open("brown_tagged.txt").read().splitlines()
        for sent in sents:
            words = sent.split()
            for word in words:
                m = re.search('_(.*)',word)
                self.tags.append(m.group(1))
        self.tags_set = set(self.tags)
    def init_tokens_tags(self):
        tagged_sents = open("brown_tagged.txt").read().splitlines()
        tags_tokens = []
        for tagged_sent in tagged_sents:
            tagged_words = tagged_sent.split()
            for tagged_word in tagged_words:
                m = re.search('(.*)_(.*)',tagged_word)
                tags_tokens.append((m.group(2),m.group(1)))
            self.tags_tokens.append(tags_tokens)
            tags_tokens = []
    def init_cfd_tagswords(self):
        self.cfddist_tagswords = CFD(j for i in self.tags_tokens for j in i)
    def init_cpd_tagswords(self):
        self.cpddist_tagswords = CPD(self.cfddist_tagswords,MLEProbDist)
    def hmm_trigram(self):
        trigrams = ngrams(self.tags,3)
        self.trigrams_as_bigrams.extend([((tags[0],tags[1]),tags[2]) for tags in trigrams])
        self.cfddist_tags = CFD(self.trigrams_as_bigrams)
        self.cpddist_tags = CPD(self.cfddist_tags,MLEProbDist)
    def hmm_test(self):
        sent = "START START The grand jury commented. STOP"
        tag_sequence = "START START AT JJ NN VBD . STOP"
        prob_tags = 1.0
        prob_tagswords = 1.0
        words_tokens = [j for i in word_tokenize(sent) for j in i]
        tags_tokens = [j for i in word_tokenize(tag_sequence) for j in i]
        for i in range(2,len(tags_tokens)):
            prob_tags*=self.cpddist_tags[(tags_tokens[i-2],tags_tokens[i-1])].prob(tags_tokens[i])
        for i,j in zip(words_tokens,tags_tokens):
            prob_tagswords*=(self.cpddist_tagswords[j].prob(i))
        prob_tot = prob_tags*prob_tagswords
        print ('The probability of "START START The grand jury commented. STOP" having tags "START START AT JJ NN VBD . STOP" is: ',prob_tot)

if __name__=="__main__":
    pos_tag = pos_tagger()
    pos_tag.tokenize()
    pos_tag.init_tags()
    pos_tag.init_tokens_tags()
    pos_tag.init_cfd_tagswords()
    pos_tag.init_cpd_tagswords()
    pos_tag.hmm_trigram()
    pos_tag.hmm_test()
