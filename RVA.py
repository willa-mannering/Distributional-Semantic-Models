#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:22:35 2018

@author: willamannering
"""

from scipy import sparse
import numpy as np
from re import sub
from nltk.corpus import stopwords
import pickle
#make sure to download stopwords
sw = set(stopwords.words("english"))

#%%

class RVA:
    
    #comparison: 1 = greater than equal; 0 = less than
    #value: threshold if kind 1; attribute index if kind 0
    #kind: 1 = value, 0 = attribute

    
    #create RVA object = object that allows user to train different models from input text
    def __init__(self, distribution = 'sb', dims = 10, sparsity = .2):
        self.memoryVectors = sparse.csr_matrix((0,0)) #create compressed sparse row matrix representing memory vectors for words
        self.dictionary = {} #initialize dictionary for words, index pairs
        self.frequency = {} #initialize dictionary to hold word, frequency pairs
        self.structure = 'rv' #set structure = random vector
        
        if(distribution == 'sb'): #create environment vector using randomly generated values from sparse binary distribtion
            self.distribution = 'sb'
            self.contextVectors = sparse.csr_matrix((0,dims))
            self.dims = dims
            self.sparsity = sparsity #set amount of sparsity
            
        elif(distribution == 'gaussian'): 
            self.distribution = 'gaussian'
            self.contextVectors = sparse.csr_matrix((0,dims))
            self.dims = dims
                

 ################ Internal Methods############################3
      
    def updateDict(self, words):
        self.dictionary = dict(zip(words, range(0, len(words))))

        if(self.distribution == 'sb'):
            #update contextVectors
            row = np.array(np.repeat(np.arange(1), self.dims*self.sparsity))
            # no repeating values within a row
            col = np.random.choice(self.dims,int(self.dims * self.sparsity*len(words)), replace=False)
            data = np.repeat(1, len(row)*len(col))
            # TODO: ensure unique row vectors
            self.contextVectors = sparse.csr_matrix((data, (row, col)), shape=(len(words),self.dims))

            #update memoryVectors
            self.memoryVectors = sparse.csr_matrix((len(words),self.dims))

        elif(self.distribution == 'gaussian'):
            
            #update contextVectors
            row = np.repeat(np.arange(len(words)),self.dims)
            # no repeating values within a row
            col = np.repeat(np.arange(self.dims), len(words))
            #data = np.random.normal(loc=0,scale=1,size=self.dims)
            data = np.random.normal(loc=0,scale=1/np.sqrt(self.dims),size=self.dims*len(words))
            # TODO: ensure unique row vectors
            self.contextVectors = sparse.csr_matrix((data, (row, col)), shape=(len(words),self.dims))
            #update memoryVectors
            self.memoryVectors = sparse.csr_matrix((len(words),self.dims))
        
        
############################ External Methods ##################################
    #read corpus based on line breaks
    
    def read(self, pathToCorpus,  accumulateFunction,  context = 'sen') :
#        if saved:
#            with open(pick, 'rb') as f:
#                corpus = pickle.load(f) 
        corpus = loadCorpus(pathToCorpus, context) #what is this context variable? 
            
        self.corpus = corpus
        c = set()
        for corp in corpus:
            c=c.union(set(corp))
            
        self.updateDict(c)
        #read corpus
        for text in corpus: #text = context sentence/paragraph
            if(len(text) <=1):
            #    print('skipped')
                continue   
            # ensure word represented in dictionary/context vectors           
            accumulateFunction(self, text)

    def read_saved(self, pathToCorpus,  saved_corp_path, accumulateFunction, context = 'sen') :

        with open(saved_corp_path, 'rb') as f:
            corpus = pickle.load(f) 
            
        self.corpus = corpus
        c = set()
        for corp in corpus:
            c=c.union(set(corp))
            
        self.updateDict(c)
        #read corpus
        for text in corpus: #text = context sentence/paragraph
            if(len(text) <=1):
            #    print('skipped')
                continue   
            # ensure word represented in dictionary/context vectors           
            accumulateFunction(self, text)

        
    #save memory and context vectors
    def save(self,path,filename):
        import pickle
        with open(path+filename+'.ph', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print('Saved.')
    
    #load memory and context vectors from pickle file    
    def load(path, filename):
        import pickle
        # Load object
        with open(path+filename+'.ph', 'rb') as input:
            test1 = pickle.load(input)
        print(filename + ' loaded.')
        return test1
    

    def getIndices(self, wordList):
        # ensure all words are in dictionary
        wordList = [i for i in wordList if i in self.dictionary.keys()]
        return [self.dictionary[i] for i in wordList]
    
   
        
class accumulate(): 
    pass
    # accumulate for random vector matrix
   
    
    def randomVector(ph, context): #ph = model object
        t = [ph.dictionary[x] for x in context] #is t the context vector?
        v = ph.contextVectors[t].nonzero()[1]
        col = np.tile(v,len(t))
        row = np.repeat(t,len(v))
        data = np.ones((len(row),),dtype=int)
        z = sparse.csr_matrix( (data,(row,col)), shape = ph.memoryVectors.shape) #two variables named t?

        ph.memoryVectors = ph.memoryVectors + z



def filter_stopwords(corpus):
    if corpus in sw:
        return False
    else:
        return True

    
def loadCorpus(filepath, context = 'sen'):
        #load corpus
        corpus = []
        ofile = open(filepath, 'r')
        for line in ofile:
            #remove symbols
            line = sub('\n', '', line)
            line = sub('\'', '', line)
            line = sub('"', '', line)
            line = sub('-', '', line)
            line = sub('\(', '', line)
            line = sub('\)', '', line)
            line = sub(',', '', line)
            line = sub(':', '', line)
            line = line.lower()
            
            if(context == 'sen'):
                line = line.split('.')
                f = line
                line = []
                [line.extend(i.split('?')) for i in f]
                f = line
                line = []
                [line.extend(i.split('!')) for i in f]
                f = line
                line = []
                [line.append(i.split(' ')) for i in f]
                line = [list(filter(None, l)) for l in line]
                line = list(filter(None,line))
                corpus.extend(line)
                
        ofile.close()
        if(isinstance(context,int)):
            f = []
            i = 0
            while(i+context <= len(corpus[0])):
                f.append(corpus[0][i:i+context])
                i = i + 1
            corpus = f

        finalCorpus = []
        for c in corpus:
            finalCorpus.append(list(filter(filter_stopwords, c)))

      
        return finalCorpus
    
    
    
def loadCorpus_saved(filepath, tosave_path, context = 'sen'):
        #load corpus
        corpus = []
        ofile = open(filepath, 'r')
        for line in ofile:
            #remove symbols
            line = sub('\n', '', line)
            line = sub('\'', '', line)
            line = sub('"', '', line)
            line = sub('-', '', line)
            line = sub('\(', '', line)
            line = sub('\)', '', line)
            line = sub(',', '', line)
            line = sub(':', '', line)
            line = line.lower()
            
            if(context == 'sen'):
                line = line.split('.')
                f = line
                line = []
                [line.extend(i.split('?')) for i in f]
                f = line
                line = []
                [line.extend(i.split('!')) for i in f]
                f = line
                line = []
                [line.append(i.split(' ')) for i in f]
                line = [list(filter(None, l)) for l in line]
                line = list(filter(None,line))
                corpus.extend(line)

#              
        ofile.close()
        if(isinstance(context,int)):
            f = []
            i = 0
            while(i+context <= len(corpus[0])):
                f.append(corpus[0][i:i+context])
                i = i + 1
            corpus = f

        finalCorpus = []
        for c in corpus:
            finalCorpus.append(list(filter(filter_stopwords, c)))


        with open(tosave_path, 'wb') as f:
            pickle.dump(finalCorpus, f)

    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    