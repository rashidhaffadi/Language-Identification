# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""

from fastcore.xtras import *
from fastcore.script import *
from text import *

@call_parse
def stop_words_intersections(max_words:Param("lenght of text used for each language.", int)=None,
                             n_jobs:Param("number of workers for parallel or threaded processing.", int)=1,
                             thread:Param("use threading instead of parallel processing.", store_true)=False, 
                             progress:Param("show progress bar.", store_true)=False,
                             path:Param("path containing text files for each language.", str)="F:/workspace/PFE/Datasets/europarl/new/",
                             dst:Param("save stop words to file.", str)="./stop_words.txt",
                             r:Param("return list of stop words", store_true)=False):

    "detect and store stop words: words that are shared between atleast two languages."

        # usage: stop_words.py [-h] [--max_words MAX_WORDS] [--n_jobs N_JOBS] [--thread]
        #                      [--progress] [--path PATH] [--dst DST] [--r] [--pdb]
        #                      [--xtra XTRA]

        # detect and store stop words: words that are shared between atleast two
        # languages.

        # optional arguments:
        #   -h, --help            show this help message and exit
        #   --max_words MAX_WORDS
        #                         lenght of text used for each language.
        #   --n_jobs N_JOBS       number of workers for parallel or threaded processing.
        #                         (default: 1)
        #   --thread              use threading instead of parallel processing.
        #                         (default: False)
        #   --progress            show progress bar. (default: False)
        #   --path PATH           path containing text files for each language.
        #                         (default: F:/workspace/PFE/Datasets/europarl/new/)
        #   --dst DST             save stop words to file. (default: ./stop_words.txt)
        #   --r                   return list of stop words (default: False)
        #   --pdb                 Run in pdb debugger (default: False)
        #   --xtra XTRA           Parse for additional args (default: '')

    texts = read_texts(labels, max_words, n_jobs, path, thread, progress)

    stp_words = []
    txts = []
    for text in texts:
        txts.append(set(text.split()))
        
    for i in range(len(txts)-1):
        for j in range(i+1, len(txts)):
            # print(i, j)
            s0 = txts[i]
            s1 = txts[j]
            stp_words.append(s0.intersection(s1))
            
    stp_words = flatten([list(s) for s in stp_words if not is_empty(s)])
    stp_words = list(set(stp_words))
    with open(dst, "wt", encoding="utf-8") as f:
        f.write(merge(stp_words))
    if r: return stp_words