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
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Collection
from fastprogress.fastprogress import progress_bar
import concurrent

# %% Global Variables
TOKENIZER_REGEX = re.compile(r"[^\d\W]+", re.UNICODE)
PAD_INDEX = 1
UNK_INDEX = 0
MIN_NGRAM_FREQ = 2
TOKEN_PREFIX = '<'
TOKEN_SUFFIX = '>'

# %% Tokenizer Class
# inheret DaseTokenizer: from fastai.text.transform import BaseTokenizer
# class SpecialTokenizer(BaseTokenizer):
class SpecialTokenizer():
    def __init__(self, vocab=None, chars_ngram_range=(2,4), chunk_size=2500, max_words=15, max_ngrams=4, words_tokenizer=None, n_jobs=1):
        self.words_tokenizer = words_tokenizer or _default_tokenizer
        self.chars_ngram_range = chars_ngram_range
        self.chunk_size = chunk_size
        self.max_words = max_words
        self.max_ngrams = max_ngrams
        self.vocab = vocab or Vocab()
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

    def fit(self, texts, chunk_size=None, n_jobs=None, bs=1):
        n_jobs = n_jobs or self.n_jobs or 1
        self.chunk_size = chunk_size or self.chunk_size
        
        print("===============Fitting Tokenizer===============")
        text = self.merge(texts)
        self.num_iters(text)
        print("Number of Iterations: "+ str(self.iters))
        print("Chunk Size: "+str(self.chunk_size))
        self.fit_tokens_stats(text, n_jobs, bs)
        self.build_vocab()
        self.ngrams_counter = Counter()
        self.fitted = True
        print("vocab size: %d" % self.vocab.get_size())
        print("===============Training Tokenizer Finished===============")
        return self
    
    def num_iters(self, text):
        it = 0
        for c in _chunkenize(text, self.chunk_size):
            it += 1
        self.iters = it
     

    def fit_tokens_stats(self, text, n_jobs=1, bs=1):
        print("fitting tokenizer with {} processes and batch size {}.".format(n_jobs, bs))
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
        print("vectorizing {} of rows ends in: {}s".format(len(texts), round(end-begin, 2)))
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
    
    def tokenize(self, texts, n_jobs=4, bs=64):
        if isinstance(texts, list) | isinstance(texts, pd.core.series.Series):
            if isinstance(texts, pd.core.series.Series):
                texts = texts.tolist()
            return  Parallel(n_jobs=n_jobs, batch_size=bs, require="sharedmem")(delayed(self.tokenize_single)(text) for text in texts)
        elif isinstance(texts, str):
            return [self.tokenize_single(texts)]
        else: raise TypeError
        #return list(map(self.tokenize_single, texts))

    def build_vocab(self):
        self.vocab = Vocab({"__PAD__":PAD_INDEX})
        for ngram, cnt in self.ngrams_counter.most_common():
            if cnt < MIN_NGRAM_FREQ: break
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

   
# %% must be in vocab
    # def convert_token_to_ids(self, token):
    #     return self.vocab.token2id(token)
    #     # return list(map(self.convert_ngrams_to_ids, token))

    # def convert_ngrams_to_ids(self, ngrams):
        
    #     if not self.vocab:
    #         raise Exception('SeqToLangVectorizer not fitted')

    #     ngram_ids = []
    #     for gram in ngrams:
    #         gram_id = self.vocab.get(gram)
    #         if gram_id: ngram_ids.append(gram_id)
    #     return ngram_ids
    
# %%
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
def _default_tokenizer(text):
    return TOKENIZER_REGEX.findall(text.lower().strip())


def _chunkenize(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def ifnone(a,b):
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def parallel(func, arr:Collection, max_workers:int=None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers<2: results = [func(o,i) for i,o in progress_bar(enumerate(arr), total=len(arr), leave=leave)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            results = []
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr), leave=leave): 
                results.append(f.result())
    if any([o is not None for o in results]): return results
        
def main():
    data0 = read_dataframe("data0")
    data = read_dataframe("data")

    t = SpecialTokenizer(chunk_size=50000)
    t.fit(data0.iloc[:, 0],  n_jobs=1, bs=4096)
    t.to_pickle("F:/workspace/PFE/Models/SeqToLang/tokenizer.pkl")
    vectors = t.vectorize(data.iloc[:10, 0], n_jobs=-1, bs=512)
    print("vectors with shape {} is created".format(vectors.shape))
    

if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    print("tokenization completing in %fs" % round(end-begin, 2))
    