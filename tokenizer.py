# tokenizer

# %% Imports
from joblib import Parallel, delayed
from collections import Counter
from tqdm import tqdm
import numpy as np
import re, itertools
from vocab import *
from text import read_dataframe
import pickle
import pandas as pd
import time
from fastcore.script import *
from fastcore.xtras import *

# %% Global Variables
TOKENIZER_REGEX = re.compile(r"[^\d\W]+", re.UNICODE)
PAD_INDEX = 1
UNK_INDEX = 0
TOKEN_PREFIX = '<'
TOKEN_SUFFIX = '>'
DEBUG = True


class SpecialTokenizer():
    def __init__(self, vocab=None, chars_ngram_range=(2,4), chunk_size=2500, min_freq=10, max_words=15, max_ngrams=4, words_tokenizer=None, n_jobs=1):
        self.words_tokenizer = words_tokenizer or _words_tokenizer
        self.chars_ngram_range = chars_ngram_range
        self.chunk_size = chunk_size
        self.max_words = max_words
        self.max_ngrams = max_ngrams
        self.vocab = vocab or Vocab()
        self.min_freq = min_freq
        self.ngrams_counter = Counter()
        self.iters = 0
        self.n_jobs = n_jobs
        self.fitted = False

        
    def merge(self, texts):
        if isinstance(texts, str):
            return texts
        res = ""
        for text in texts:
            res = res +" "+ text.strip()
        return res

    def num_iters(self, text):
        it = 0
        for c in _chunkenize(text, self.chunk_size):
            it += 1
        self.iters = it
        return it

    def fit(self, texts, chunk_size=None, n_jobs=None, bs=1, r=False):
        n_jobs = n_jobs or self.n_jobs or 1
        self.chunk_size = chunk_size or self.chunk_size
        
        if DEBUG: print("===============Fitting Tokenizer===============")
        text = self.merge(texts)
        self.num_iters(text)
        if DEBUG: print("Number of Iterations: "+ str(self.iters))
        if DEBUG: print("Chunk Size: "+str(self.chunk_size))
        self.update_vocab(text, n_jobs, bs)
        self.build_vocab()
        self.ngrams_counter = Counter()
        self.fitted = True
        if DEBUG: print("vocab size: %d" % self.vocab.get_size())
        if DEBUG: print("===============Training Tokenizer Finished===============")
        if r: return self

    def update_vocab(self, text, n_jobs=1, bs=1, threadpool=False):
        if DEBUG: print("fitting tokenizer with {} processes and batch size {}.".format(n_jobs, bs))
        if n_jobs == 1:
            for chunk in tqdm(_chunkenize(text, self.chunk_size), desc="Progress", total=self.iters):
                tokenized_text = self.tokenize_single(chunk)
                for tokens in tokenized_text:
                    if len(tokens) != 0:
                        self.ngrams_counter.update(tokens)
        else:
            # change this shit
            # must add security mesure about the lenght of text and the number of jobs
            tot = self.iters/n_jobs if n_jobs>0 else self.iters/4
            for big_chunk in tqdm(_chunkenize(text, self.chunk_size*n_jobs if n_jobs>0 else self.chunk_size*4), total=tot, desc="Progress"):
                # tokens = parallel(self.tokenize_single, _chunkenize(big_chunk, self.chunk_size), n_workers=n_jobs, threadpool=threadpool, chunksize=bs).items
                tokens = Parallel(n_jobs=n_jobs, batch_size=bs, require="sharedmem")(delayed(self.tokenize_single)(chunk) for chunk in _chunkenize(big_chunk, self.chunk_size))
                tokens = list(itertools.chain.from_iterable(tokens))
                tokens = list(itertools.chain.from_iterable(tokens))
                self.ngrams_counter.update(tokens)
                
    def vectorize(self, texts, n_jobs=1, bs=64):
        begin = time.time()
        tokenized_texts = self.tokenize(texts, n_jobs, bs) # rank 3
        vectors = self.vocab.tokens2ids(tokenized_texts, n_jobs)
        # vectors = list(map(self.convert_token_to_ids, tokenized_texts))
        padded_texts = self.pad_tokenized_texts(vectors)
        padded_texts = np.array(padded_texts, dtype=np.int)
        end = time.time()
        if DEBUG: print("vectorizing {} of rows ends in: {}s".format(len(texts), round(end-begin, 2)))
        return padded_texts
    
    def word2ngrams(self, token):
        token = TOKEN_PREFIX + token + TOKEN_SUFFIX
        ngrams = set()
        for ngram_size in range(self.chars_ngram_range[0], self.chars_ngram_range[1]+1):
            grams = zip(*[token[i:] for i in range(ngram_size)])
            ngrams |= {"".join(g) for g in grams}

        return list(ngrams)

    def tokenize_single(self, text):
            words = self.words_tokenizer(text)
            tokens = list(map(self.word2ngrams, words))
            return list(tokens)
    
    def tokenize(self, texts, n_jobs=4, bs=64, threadpool=False):
        if isinstance(texts, list) | isinstance(texts, pd.core.series.Series):
            if isinstance(texts, pd.core.series.Series):
                texts = texts.tolist()
            return parallel(self.tokenize_single, texts, n_workers=n_jobs, threadpool=threadpool, chunksize=bs).items
            # return  Parallel(n_jobs=n_jobs, batch_size=bs, require="sharedmem")(delayed(self.tokenize_single)(text) for text in texts)
        elif isinstance(texts, str):
            return [self.tokenize_single(texts)]
        else: raise TypeError
        #return list(map(self.tokenize_single, texts))

    def build_vocab(self, min_freq=None):
        min_freq = min_freq if min_freq is not None else self.min_freq
        self.vocab = Vocab({"__PAD__":PAD_INDEX})
        for ngram, cnt in self.ngrams_counter.most_common():
            if cnt < min_freq: break
            if ngram not in {TOKEN_PREFIX, TOKEN_SUFFIX}:
                self.vocab.update(ngram, self.vocab.get_size())
        self.vocab.fitted()     

    def pad_tokenized_texts(self, tokenized_texts):
        padded = []
        for tokens in tokenized_texts:
            padded_tokens = list(map(self.pad_single_token, tokens))
            padded.append(padded_tokens[:self.max_words] + [np.zeros(self.max_ngrams)]*(self.max_words - len(padded_tokens)))
        return padded

    def pad_single_token(self, token):
        return token[:self.max_ngrams] + [0]*(self.max_ngrams - len(token))

    @classmethod
    def from_pickle(self, f):
        """
        Parameters
        ----------
        f : path to file
            the file name and path in which a vocab object is saved.

        Returns
        -------
        Vocab
        return a Vocab object from a pickle file.

        """
        with open(f, "rb") as fp:
            return pickle.load(fp)
    
    def to_pickle(self, f):
        """
        Parameters
        ----------
        f : path to a file
            the file name and path in which we want to save our Vocab object.

        Returns
        -------
        None.

        """
        self.f = True
        with open(f, "wb") as fp:
            pickle.dump(self, fp)
    



# %%
def _words_tokenizer(text): # replace spacy
    return TOKENIZER_REGEX.findall(text)


def _chunkenize(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
      
@call_parse  
def create_tokenizer(dataset:Param("name of a dataset.", str)="data0",
                     chunk_size:Param("size of text treated each time.", int)=100000,
                     n_jobs:Param("number of processes.", int)=1,
                     bs:Param("number of chunks treated in parallel.", int)=4096,
                     min_freq:Param("drop evary token with frequency less than 'min_freq'.", int)=10,
                     save:Param("save tokenizer to pickle", str)="F:/workspace/PFE/Models/SeqToLang/tokenizer.pkl",
                     r:Param("return tokenize", store_true)=False):

        # usage: tokenizer.py [-h] [--dataset DATASET] [--chunk_size CHUNK_SIZE]
        #                     [--n_jobs N_JOBS] [--bs BS] [--save SAVE] [--r] [--pdb]
        #                     [--xtra XTRA]

        # optional arguments:
        #   -h, --help            show this help message and exit
        #   --dataset DATASET     (default: data0)
        #   --chunk_size CHUNK_SIZE
        #                         (default: 50000)
        #   --n_jobs N_JOBS       (default: 1)
        #   --bs BS               (default: 4096)
        #   --save SAVE           save tokenizer to pickle (default:
        #                         F:/workspace/PFE/Models/SeqToLang/tokenizer.pkl)
        #   --r                   return tokenize (default: False)
        #   --pdb                 Run in pdb debugger (default: False)
        #   --xtra XTRA           Parse for additional args (default: '')

    begin = time.time()
    data0 = read_dataframe(dataset)
    # data = read_dataframe("data")

    t = SpecialTokenizer(chunk_size=chunk_size, min_freq=min_freq)
    t.fit(data0.iloc[:, 0],  n_jobs=n_jobs, bs=bs)
    if save != "": t.to_pickle(save)
    # vectors = t.vectorize(data.iloc[:10, 0], n_jobs=-1, bs=512)
    # if DEBUG: print("vectors with shape {} is created".format(vectors.shape))
    end = time.time()
    if DEBUG: print("tokenization completing in %fs" % round(end-begin, 2))
    if r: return t

    
    
    