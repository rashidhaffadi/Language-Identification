
# helper functions for text processing
# %% imports
import re
from path import Path
import pandas as pd
import numpy as np
from functools import partial
from joblib import Parallel, delayed
import random, sys, time

DEBUG = False

# %% global variables
languages = ['Bulgarian', 'Czech', 'Danish', 'German', 'Greek', 'English', 'Spanish', 'Estonian', 
             'Finnish', 'French', 'Hungarian', 'Italian', 'Lithuanian', 'Latvian', 'Dutch', 
             'Polish', 'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Swedish']

labels=['bg','cs','da','de','el','en','es','et','fi','fr',
        'hu','it','lt','lv','nl','pl','pt','ro','sk','sl','sv']

# %% helper functions
""" get path for each language in the europrl dataset
lang: language code
"""
def get_path(lang):
    path = Path("F:\\workspace\\PFE\\Datasets\\europarl\\txt")
    path = path/lang
    return path

"""read text from europrl dataset
lang: language code
max_len: max number of words
"""

def get_text(lang, max_len=50000):
    path = get_path(lang)
    text = ""
    for f_name in path.listdir():
        with open(f_name, encoding="utf-8") as f:
            text = text + " " + f.read()
        if len(text.split(" ")) >= max_len:
            break 
    text = " ".join(text.split(" ")[:max_len])# ensure the class balance
    return text

"""
"""

def remove_tags(text):
    return re.sub(r'<(!?).*>', ' ', text)

"""
"""
def remove_numerals(text):
    return re.sub(r'\d+', ' ', text)

"""
"""
def remove_endl(text):
    return re.sub(r'\n', ' ', text)

"""
"""
def remove_punct(text):
    return re.sub(r'[,\.\*;:!?\"\'-\[\]{}&/$]+', ' ', text)

"""
"""
def remove_extra_space(text):
    return re.sub(r' [ ]+', ' ', text)

"""
"""
def remove_forein(text):
    return re.sub(r'\(.*\)', ' ', text)
    
"""
"""
def lower(text):
    return text.lower()

"""
"""
def strip(text):
    return text.strip()

""" preprocess text
- remove tags
- remove numerical values, "\n"
- remove punctuations

"""
def preprocess(text):
    text = lower(text)
    text = remove_tags(text)
    text = remove_forein(text)
    text = remove_numerals(text)
    text = remove_endl(text)
    text = remove_punct(text)
    text = remove_extra_space(text)
    text = strip(text)
    return text
# """
# """
# def listToText(str_list): 
#     string = ' '.join(str_list)
#     return(string)

"""
"""
def lang_text(lang, max_words=50000):
    path = get_path(lang)
    text = get_text(path, max_words)
    text = preprocess(text)
    return text

"""save the text corresponding to {lang} in a txt file
max_words: maximum number of words in the text file
"""
def save_text(lang, max_words=50000, text=None):
    path = "F:/workspace/PFE/Datasets/europarl/new/" + lang + ".txt"
    
    with open(path, "wt", encoding="utf-8") as f:
        if text is not None:
            f.write(text)
        else:
            text = lang_text(lang, max_words)
            f.write(text)
    if DEBUG: print("{} text saved in: {}".format(languages[labels.index(lang)], path))       

"""save texts in different files
"""
def save_texts(langs, max_words=50000, texts=dict(), n_jobs=None):
    
    if n_jobs is None or n_jobs == 1:
        if DEBUG: print("creating dataset files using one process .....")
        for lang in langs:
            if texts and texts.get(lang) is not None:
                save_text(lang, max_words, texts.get(lang))
            else:
                save_text(lang, max_words)
    else:
        if DEBUG: print("creating dataset files using parallel computation with %d processes....." % n_jobs)
        f = partial(save_text, max_words=max_words, text=None)
        Parallel(n_jobs=n_jobs)(delayed(f)(lang) for lang in langs)
        
            
"""read the saved file
"""
def read_text(lang, max_words):
    with open("F:/workspace/PFE/Datasets/europarl/new/" + lang + ".txt", "rt", encoding="utf-8") as f:
        text = f.read()
        text = " ".join(text.split(" ")[:max_words])
    return text

"""
"""
def read_texts(langs, max_words, n_jobs=1):
    if n_jobs == 1:
        strings = dict()
        for lang in langs:
            strings[lang] = read_text(lang, max_words)
    else:
        f = partial(read_text, max_words=max_words)
        strings = Parallel(n_jobs=n_jobs)(delayed(f)(lang) for lang in langs)
    return strings

"""create dataframe from dictionary of texts if provided, or text files
lang: list of languages
texts: dict, dinctionary of strings - {lang:lang_text, ...} -
create_files: bool, create text file for each language before loading it into dataframe
max_words: int, 
"""
def create_dataframe(labels, texts=dict(), create_files=False, max_words=50000, n_jobs=None):
    begin = time.time()
    if create_files:
        save_texts(labels, max_words, n_jobs=n_jobs)
    strings = []
    
    if n_jobs is None or n_jobs == 1:
        if DEBUG: print("reading dataset files using one process.....")
        for label in labels:
            if texts.get(label) and texts.get(label) != "":
                strings.append(texts[label])
            else:
                strings.append(read_text(label, max_words))
    else:
        if DEBUG: print("reading dataset files using parallel processing with %d processes....." % n_jobs)
        f = partial(read_text, max_words=max_words)
        strings = Parallel(n_jobs=n_jobs)(delayed(f)(label) for label in labels)
    dataframe = pd.DataFrame(strings, columns=['text'])
    # dataframe.reset_index(inplace=True)
    dataframe["label"] = list(range(len(labels)))
    end = time.time()
    took = end - begin
    if DEBUG: print("dataframe creation with %d words ends in %fs." % (max_words, int(took)))
    return dataframe

""" split one text to multiple sentences with a fix lenght
"""
def create_sentences(text, label, max_lenght):
    if isinstance(label, str): label = labels.index(label)
    if DEBUG: print("creating sentences from {} text..... ".format(languages[label]))
    df = pd.DataFrame()
    words = text.split(" ")
    sentences = []
    len_ = len(words)
#     possible to randomize the lenght between [1...max_lenght]
    lenght = random.randint(2, max_lenght)
    i = 0
    j = i+lenght
    while True:
        sentences.append(' '.join(words[i:j]))
        i = j
        lenght = random.randint(1, max_lenght)
        j += lenght
        if j >= len_:
            break
     
    df["text"] = sentences
    df["label"] = label
    return df

"""
"""
def create_split_dataframe(df, max_lenght, n_jobs=1):
    begin = time.time()
    if n_jobs is None or n_jobs == 1:
        if DEBUG: print("spliting text to sentences for each language with maximum {} words for each language: ".format(max_lenght))
        data = pd.concat([create_sentences(text, label, max_lenght) for text, label in zip(df["text"], df["label"])], ignore_index=True)
    else:
        if DEBUG: print("spliting text to sentences for each language with maximum {} words using parallel processing with {} processes: ".format(max_lenght, n_jobs))
        f = partial(create_sentences, max_lenght=max_lenght)
        dfs = Parallel(n_jobs=n_jobs)(delayed(f)(text, label) for text, label in zip(df['text'], df['label']))
        data = pd.concat(dfs, ignore_index=True)
    data = data.sample(frac=1) #shuffle data
    end = time.time()
    t = end - begin
    if DEBUG: print("spliting dataframe to sentences ended.")
    if DEBUG: print("{} rows created in {}s".format(data.shape[0], int(t)))
    return data

def merge(texts):
        res = ""
        for text in texts:
            res = res +" "+ text
        return res.strip()
    
def trn_tst(dataframe, trn_p=0.8):
    trn_df = pd.DataFrame()
    tst_df = pd.DataFrame()
    trn_txt = []
    tst_txt = []
    for text in dataframe.text:
        lst = text.split(" ")
        len_ = len(lst)
        trn_txt.append(merge(lst[:int(trn_p*len_)]))
        tst_txt.append(merge(lst[int(trn_p*len_):]))
        
    trn_df['text'] = trn_txt
    tst_df['text'] = tst_txt
        
    trn_df['label'] = dataframe.label
    tst_df['label'] = dataframe.label
    return trn_df, tst_df


"""save dataframe
mode: f for feather, p for pickle
"""
def save_dataframe(name, dataframe, mode="p"):
    if mode == "p":
        dataframe.to_pickle("F:/workspace/PFE/Datasets/europarl/{}.pkl".format(name))
        if DEBUG: print("saving dataframe to F:/workspace/PFE/Datasets/europarl/{}.pkl.....".format(name))
    else:
        dataframe.to_feather("F:/workspace/PFE/Datasets/europarl/{}.fth".format(name))
        if DEBUG: print("saving dataframe to F:/workspace/PFE/Datasets/europarl/{}.fth.....".format(name))

"""read saved dataframe
mode: p, f same as save_dataframe
"""
def read_dataframe(name, mode="p"):
    if mode == "p":
        dataframe = pd.read_pickle("F:/workspace/PFE/Datasets/europarl/{}.pkl".format(name))
    else:
        dataframe = pd.read_feather("F:/workspace/PFE/Datasets/europarl/{}.fth".format(name))
    return dataframe

"""sample 
"""
def sample(df, label, n, balanced=True, shuffle=True):
    if shuffle:
        df = df.sample(frac=1.0)
    if balanced:
        g = df.groupby(label)
        g = g.apply(lambda x: x.sample(n).reset_index(drop=True))
    return g
    
    

"""
"""
def main(labels, max_words, max_lenght, name=None, n_jobs=1, cfiles=False):
    
    if DEBUG: print("creating dataframe with max {} words for each language.....".format(max_words))
    dataframe = create_dataframe(labels, create_files=cfiles, max_words=max_words, n_jobs=n_jobs)
    if name is not None and isinstance(name, str):
        save_dataframe(name + "0", dataframe)
    dataframe = create_split_dataframe(dataframe, max_lenght, n_jobs=n_jobs)
    dataframe.reset_index(drop=True, inplace=True)
    if name is not None and isinstance(name, str):
        save_dataframe(name, dataframe)
    if DEBUG: print(dataframe.head())
    if DEBUG: print("saved dataframe with {} rows and {} columns".format(dataframe.shape[0], dataframe.shape[1]))
    if DEBUG: print("end.")
    return dataframe


# %% main execution
if __name__ == "__main__":
    
    begin = time.time()
    if DEBUG: print("exec file: " + sys.argv[0])
    
   #  try:
   #    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
   # except getopt.GetoptError:
   #    print 'test.py -i <inputfile> -o <outputfile>'
   #    sys.exit(2)
    
    if len(sys.argv) > 1:
     	max_words = int(sys.argv[1])
    else:
    	max_words = 50000

    if len(sys.argv) > 2:
        max_lenght = int(sys.argv[2])
    else: 
        max_lenght = 25

    if len(sys.argv) > 3:
        name = sys.argv[3]
    else: 
        name = None
        
    if len(sys.argv) > 4:
        n_jobs = int(sys.argv[4])
    else: 
        n_jobs = 1

    p = 0.75
    answer = input("do you need to create new dataset files (y/n)?: ")
    answer = answer.lower()
    if answer in ["y", "yes"]:
        cfiles = True
    else:
        cfiles = False
    df = main(labels, max_words, max_lenght, name, n_jobs, cfiles)
    l = len(df)
    idxs = np.random.permutation(l)
    train_ds = df.loc[idxs[:int(p*l)]]
    valid_ds = df.loc[idxs[int(p*l)+1:]]
    train_ds.reset_index(drop=True, inplace=True)
    valid_ds.reset_index(drop=True, inplace=True)
    save_dataframe("train_ds", train_ds)
    save_dataframe("valid_ds", valid_ds)
    end = time.time()
    
    if DEBUG: print("completing in %fs" % round(end-begin, 2))
    
    
    
    
    
    