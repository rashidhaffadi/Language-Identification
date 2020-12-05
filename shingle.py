# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""
import pickle
import itertools
from exceptions import ArgumentError
from collections import Counter
from text import * #create_dataframe, read_dataframe, trn_tst
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
from fastcore.script import *
import math

f1 = partial(f1_score, average='macro')
accuracy = accuracy_score
# %% global variables
languages = ['Bulgarian', 'Czech', 'Danish', 'German', 'Greek', 'English', 'Spanish', 'Estonian', 
             'Finnish', 'French', 'Hungarian', 'Italian', 'Lithuanian', 'Latvian', 'Dutch', 
             'Polish', 'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Swedish']
labels = ['bg','cs','da','de','el','en','es','et','fi','fr',
          'hu','it','lt','lv','nl','pl','pt','ro','sk','sl',
          'sv']

# %%

class Shingle():
    def __init__(self, df=None, x="text", y="label", ngram_range=(1, 3), min_freq=10, 
                 smode="sum", pad_right=">", pad_left="<", model=dict(), metrics=[f1]):
        
        self.df = df
        self.y = y
        self.x = x
        self.scoring_mode = smode
        self.ngram_range = ngram_range
        self.pad_right, self.pad_left = pad_right, pad_left
        self.model = model
        self.metrics = metrics
        self.min_freq = min_freq
        self.fitted = True if len(model) != 0 else False
        # self.pad_right_id, self.pad_left_id = pad_right_id, pad_left_id
        
    def add_metric(self, m):
        self.metrics = self.metrics + [m]
    
    def _pad_zero(self):
        for label in self.model:
            self.model[label][self.pad_left] = 0
            self.model[label][self.pad_right] = 0
    
    def _merge_texts(self, texts):
        """merge texts to one text
        """
        if isinstance(texts, str):
            return texts
        res = ""
        for text in texts:
            res = res +" "+ text.strip()
        return res
    
    def _word2ngrams(self, word):
        if isinstance(word, str):
            word =  self.pad_left + word + self.pad_right
            ngrams = set()
            for ngram_size in range(self.ngram_range[0], self.ngram_range[1]+1):
                grams = zip(*[word[i:] for i in range(ngram_size)])
                ngrams |= {"".join(g) for g in grams}
    
            return list(ngrams)
        else: return []
     
    def _words2ngrams(self, words):
        tokens = [self._word2ngrams(word) for word in words]
        return list(itertools.chain.from_iterable(tokens))
    
    def _update(self, label, tokens):
        self.model[label].update(tokens)
        
    def _normalize(self, label):
        _sum_ = sum(self.model[label].values())
        _model_ = self.model[label]
        for k in self.model[label]:
            _model_[k] = _model_[k]/_sum_
        self.model[label] = _model_
        
# %% Model Training        
    def _fit_one(self, label, text, bs):
        self.model[label] = Counter()
        words = self._merge_texts(text).split()
        for chunk in _chunkenize(words, self.bs):
            tokens = self._words2ngrams(chunk)
            self._update(label, tokens)
        self.model[label] = dict(self.model[label])
        self.model[label] = {k:v for k,v in zip(self.model[label].keys(), self.model[label].values()) if v > self.min_freq}
        self._normalize(label)

    def fit(self, df=None, bs=32, min_freq=None):
        min_freq = min_freq if min_freq is not None else self.min_freq
        if df is not None:
            self.df = df.copy()
        if self.df is None: raise ArgumentError("df", "Shingle.fit", 
                                                None, "pandas.DataFrame")
        self.bs = bs
        for label, text in zip(self.df.loc[:, self.y], self.df.loc[:, self.x]):
            self._fit_one(label, text, bs)
        self._pad_zero()
        self.fitted = True
        return self
# %% Model Prediction    
    def _score(self, text, label, mode=None): # could be changed if the scoring method is changed
        # using model of {label} return a score of {text}
        
        score = 0
        tokens = self._words2ngrams(text.split())
        mode = mode if mode is not None else self.scoring_mode
        self.scoring_mode = mode

        if mode == "sum":
            for token in tokens:
                tmp = self.model[label].get(token) or 0
                if tmp < 0 : tmp = -tmp
                score += tmp
        elif mode == "vote":
            for token in tokens:
                tmp = (1 if self.model[label].get(token) else 0)
                score += tmp
        elif mode == "average":
            for token in tokens:
                tmp = self.model[label].get(token) or 0
                if tmp < 0 : tmp = -tmp
                score += tmp
            score /= len(tokens)
        elif mode == "product":
            score = 1
            for token in tokens:
                if self.model[label].get(token) is not None: tmp = self.model[label].get(token) + 1
                else: tmp = 1
                if tmp < 0 : tmp = -tmp
                score *= tmp
        elif mode == "log":
            score = 1
            for token in tokens:
                if self.model[label].get(token) is not None: 
                    tmp = self.model[label].get(token) + 1
                else: 
                    tmp = 1
                if tmp < 0 : tmp = -tmp
                score *= tmp
            score = math.log10(score)   
        return {label:score}

    def _scores(self, text):
        labels = self.df.loc[:, self.y].tolist()
        scores = {}
        for label in labels:
            scores.update(self._score(text, label))
        return scores

    def _predict_one(self, text):
        scores = self._scores(text)
        if sum(scores.values()) != 0:      
            return max(scores, key=lambda k: scores[k])
        else:
            return "unk"

    def predict(self, texts):
        
        predictions = [self._predict_one(text) for text in texts]
        labels = self.df[self.y].tolist()
        labels += ['unk']
        predictions = [labels.index(p) for p in predictions]
        return predictions
    
# %% Model Evaluation
    def evaluate(self, x_test, y_test):
        y_preds = self.predict(x_test)
        return [metric(y_preds, y_test) for metric in self.metrics]
    
# %% Model Summary
    
    def summary(self):
        if self.fitted:
            df = pd.DataFrame(columns=['suported language', 'tokens count', 'frequent tokens'])
            df['suported language'] = labels = self.df.loc[:, self.y].tolist()
            df['tokens count'] = [len(self.model[label]) for label in labels]
            df['frequent tokens'] = [", ".join([t for t in sorted(self.model[label], 
                                                                  key= lambda k: self.model[label][k], 
                                                                  reverse=True)[2:7]+['...']]) 
                                     for label in labels]
            print(df)
            print("This model support %d languages." % len(labels))
            print("The scoring method used is: {%s}." % self.scoring_mode)
            print("The n-gram range used to tokenize the text data is: {}.".format(self.ngram_range))
        else:
            print("this model is not yet compiled!!")

# %% Serialization
    @classmethod   
    def from_pickle(f):
        with open(f, "rb") as fp:
            return pickle.load(fp)
    
    def to_pickle(self, f):
        self.f = True
        with open(f, "wb") as fp:
            pickle.dump(self, fp)
        
# %%    
def _chunkenize(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

@call_parse
def main(dataset:Param("", str)=None, 
         max_words:Param("", int)=1000, 
         min_freq:Param("", int)=10, 
         bs:Param("", int)=1024, 
         to:Param("", str)="./models/shingle.pkl", 
         summary:Param("", store_true)=False, 
         mode:Param("", str)="vote", 
         xlabel:Param("", str)="text", 
         ylabel:Param("", str)="label", 
         nrange:Param("", tuple)=(1, 2), 
         max_lenght:Param("", int)=15, 
         p:Param("split dataset to p*len(dataset) examples for training and the rest for testing.", float)=0.75, 
         n_jobs:Param("", int)=1): 

    if dataset is None: data = create_dataframe(labels, max_words=max_words)
    else: data = read_dataframe(dataset)

    trn_ds, tst_ds = trn_tst(data, p)

    model = Shingle(trn_ds, min_freq=min_freq, smode=mode, x=xlabel, y=ylabel, ngram_range=nrange)
    model.fit(bs=bs)

    tst_ds = create_split_dataframe(tst_ds, max_lenght, n_jobs)
    print(f'test f1: {model.evaluate(tst_ds["text"], tst_ds.label)}')
    

    if to is not None: model.to_pickle(to)
    if summary: model.summary()
    


# if __name__ == "__main__":
    
#     # data = read_dataframe("data0")
#     labels = ['bg','cs','da','de','el','en','es','et','fi','fr','hu','it',
#               'lt','lv','nl','pl','pt','ro','sk','sl','sv']
#     data = create_dataframe(labels, max_words=1000)
#     model = Shingle(data, min_freq=10)
#     model.fit(bs=1000)
#     model.to_pickle("./models/shingle.pkl")
#     model.summary()
#     for i in range(len(labels)): print(len(model.model[i]))
    
        
    
    
    
    