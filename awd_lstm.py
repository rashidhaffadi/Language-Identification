# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""

from tokenizer import SpecialTokenizer, _words_tokenizer
from fastai import *
from fastai.text import *
from fastai.callbacks import *
# from fastai.callbacks.tensorboard import *
import gc
from fastcore.script import *
from fast_util import *

# defaults = {"epochs":5,
# 			"bs":64,
# 			"tok":"F:/workspace/PFE/Models/SeqToLang/tokenizer.pkl",

# 			}

DEBUG = True

@call_parse
def main(epochs:Param("number of epochs.", int)=5, 
		 bs:Param("batch size for dataloader.", int)=64,
		 tok:Param("tokenizer path", str)="/content/tokenizer.pkl",
		 summary:Param("print model summary.", store_true)=False,
		 save:Param("save model.", store_true)=False,
		 f1:Param("use f1 score.", store_true)=False,
		 fp16:Param("use half precision points training.", store_true)=False,
		 tb:Param("tensorboard base dir (path)", str)=None,
		 config:Param("use config file to initiate AWD_LSTM model.", str)=None,
		 train_ds:Param("path to train dataset(dataframe)", str)="/content/drive/My Drive/europarl/train_ds.pkl",
		 valid_ds:Param("path to validation dataset(dataframe)", str)="/content/drive/My Drive/europarl/valid_ds.pkl",
		 test_ds:Param("path to test dataset(dataframe)", str)=None,
		 max_vocab:Param("maximum vocab.", int)=200000,
		 n_sample:Param("sample from train and validation datasets.", int)=200000,
		 p:Param("sample from train and validation datasets (n_sample*p) fro trn and (n_sample*(1-p)) for val.", float)=0.7,
		 seed:Param("set seed for reproducible results.")=None,
		 lr:Param("learning rate", float)=0.1):

	"train an AWD_LSTM model to identify a text language"
	# read or create dataset
	# tokenize dataset
	# initiate databunch(dataset + dataloader)
	# initiate or load model
	# train model
	# evaluate model
	# results to collection{model, eval, size}
	if DEBUG: print("reading dataset")
	trn, val, tst = read_ds(train_ds, valid_ds, test_ds)

	if DEBUG: print("reading tokenizer")
	t = SpecialTokenizer.from_pickle(tok)

	if DEBUG: print("sampling")
	trn, val = sample(n_sample, trn, val, p=p, seed=seed)

	if DEBUG: print("init databunch")
	tdb = init_databunch(trn, val, t, bs=bs, max_vocab=max_vocab, path='.')

	if DEBUG: print("configure")
	config = configure("awd_lstm", config)

	metrics = [accuracy]
	if f1: 
		f1 = FBeta()
		metrics.append(f1)
	model = init_model(tdb, AWD_LSTM, model_name="awd_lstm", metrics=metrics, config=config, summary=summary)
	
	cbs = []
	if save: cbs.append(SaveModelCallback(clf, every='epoch', monitor='accuracy', name='awd_model'))
	# if tb: cbs.append(LearnerTensorboardWriter(model, base_dir=Path(tb), name="run"))
	if fp16: model = model.to_fp16()
	model = fit_model(model, epochs, lr, callbacks=cbs)

	# eval, interpret