# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 01:08:53 2020

@author: Rashid Haffadi
"""
import pickle
from joblib import Parallel, delayed


class Vocab():
    """Contain the correspondence between numbers and tokens and numericalize.
    """
    def __init__(self, v:dict=dict(), f:bool=False):
        self.v = v
        self.size = len(v)
        self.f = f
        
    def fitted(self, f:bool=True):
        self.f = f
        
    def get_vocab(self):
        return self.v
    
    def _get_tokens(self):
        """Return all tokens in the vocab
        """
        return list(self.v.keys())
    
    def set_vocab(self, v:dict):
        self.v = v
        
    def update(self, ngram, value):
        self.v.update({ngram:value})
        self.size = len(self.v)
        
    def get_size(self):
        return self.size
    
    def stoi(self, key: str):
        return self.v.get(key) or 0
    
    def itos(self, value: int):
        return [k for k in self.v if self.v[k] == value]
    
    def convert_tokens_to_ids(self, tokens):
        if not self.f:
            raise Exception('Tokenizer not fitted, vocab not constructed!!')

        tokens_ids = []
        for token in tokens:
            token_id = self.stoi(token)
            if token_id: tokens_ids.append(token_id)
        return tokens_ids
    
    def token2id(self, token):
        id_ = self.v.get(token) or 0
        return id_
    
    def _tokens2ids(self, ttext):
        words = []
        for tokens in ttext:
            tmp = []
            for token in tokens:
                tmp.append(self.stoi(token))
            words.append(tmp)
        return words
    
    def tokens2ids(self, tokenized_texts, n_jobs=4):
        """
        

        Parameters
        ----------
        tokenized_texts : Rank 3 List
            rank 3 list where the first dimension is  tokenized texts, the second 
            is words whithin each text, and the third is the ngrams for each word.


        """
        if n_jobs == 1:
            vectors = [self._tokens2ids(ttext) for ttext in tokenized_texts]
        else:
            vectors = Parallel(n_jobs=n_jobs, batch_size=512)(delayed(self._tokens2ids)(ttext) for ttext in tokenized_texts)
        return vectors
                    
    
    # @staticmethod
    def from_pickle(self, f):
        with open(f, "rb") as fp:
            return pickle.load(fp)
    
    def to_pickle(self, f):
        self.f = True
        with open(f, "wb") as fp:
            pickle.dump(self, fp)

    
            
            

        
        
        
        
        
        
        
        