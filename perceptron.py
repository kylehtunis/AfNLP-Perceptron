#!/usr/bin/env python3
"""
ANLP A4: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob

from collections import Counter
from math import log
from numpy import mean
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

#only for testing
import winsound

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""


    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        # open the file at file_path, construct a list of its word tokens,
        # and append that list to 'docs'.
        # look up the document's label and append it to 'labels'.
        file=open(file_path)
        docs.append(file.read().split())
        file.close()
        labels.append(labelMap[filename])
    return docs, labels

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    
    ff[doc[0]]=1
    prev=doc[0]
    ff[doc[1]]=1
    prev2=doc[1]
    #bias feature
    ff['bias']=1
    for word in doc[2:]:
        
        #case normalization
        word=word.lower()
        
        #lemmatization
        word=WordNetLemmatizer().lemmatize(word)
        
        #binary unigram
        if ff[word]==0:
            ff[word]=1
        
        #word trigram counts
        ff[prev2+' '+prev+' '+word]+=1
        prev2=prev
        prev=word
    
    return ff

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        for iteration in range(self.MAX_ITERATIONS):
            updates=0
            correct=0
            total=0.
            for i in range(len(train_docs)):
                gold=train_labels[i]
                doc=train_docs[i]
                maxResult=-1.
                pred=''
                for c in self.CLASSES:
                    result=sum(self.weights[c][k]*doc[k] for k in doc)
                    if result>maxResult:
                        maxResult=result
                        pred=c
                if pred!=gold:
                    self.weights[gold]+=doc
                    self.weights[pred]-=doc
                    updates+=1
                    correct-=1
                correct+=1
                total+=1
            print('Iteration '+str(iteration+1)+': updates='+str(updates)+', train accuracy='+str(correct/total)+', dev accuracy='+str(self.test_eval(self.dev_docs, self.dev_labels)))
            if total==correct:
                print('Converged after '+str(iteration+1)+' iterations')
                break
            
                    

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        return ...

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        maxResult=-1.
        pred=''
        for c in self.CLASSES:
            result=sum(self.weights[c][k]*doc[k] for k in doc)
            if result>maxResult:
                maxResult=result
                pred=c
        return pred

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()


if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])

    train_docs, train_labels = load_featurized_docs('train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc = ptron.test_eval(test_docs, test_labels)
    print(acc, file=sys.stderr)
    
    #for testing
    winsound.Beep(440, 500)