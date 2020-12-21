# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""

from tokenizer import SpecialTokenizer, _words_tokenizer
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from fastai.callbacks.tensorboard import *
import gc
# from fastcore.script import *
from fastscript import *
from fast_util import *
store_true = bool_arg

@call_parse
def main(epochs:Param("number of epochs.", int)=5, 
		 bs:Param("batch size for dataloader.", int)=64,
		 tok:Param("tokenizer path", str)="/content/tokenizer.pkl",
		 summary:Param("print model summary.", store_true)=False,
		 save:Param("save model.", store_true)=False,
		 f1:Param("use f1 score.", store_true)=False,
		 fp16:Param("use half precision points training.", store_true)=False,
		 tb:Param("tensorboard base dir (path)", str)=None,
		 config:Param("use config file to initiate Transformer model.", str)=None,
		 train_ds:Param("path to train dataset(dataframe)", str)="/content/train_ds.pkl",
		 valid_ds:Param("path to validation dataset(dataframe)", str)="/content/valid_ds.pkl",
		 test_ds:Param("path to test dataset(dataframe)", str)=None,
		 max_vocab:Param("maximum vocab.", int)=200000,
		 n_sample:Param("sample from train and validation datasets.", int)=200000,
		 p:Param("sample from train and validation datasets (n_sample*p) fro trn and (n_sample*(1-p)) for val.", float)=0.7,
		 seed:Param("set seed for reproducible results.")=None,
		 lr:Param("learning rate", float)=0.1):

	"train a Transformer model to identify a text language"
	# read or create dataset
	# tokenize dataset
	# initiate databunch(dataset + dataloader)
	# initiate or load model
	# train model
	# evaluate model
	# results to collection{model, eval, size}
	
	trn, val, tst = read_ds(train_ds, valid_ds, test_ds)

	t = SpecialTokenizer.from_pickle(tok)
	trn, val = sample(n_sample, trn, val, p=p, seed=seed)

	tdb = init_databunch(trn, val, t, bs=bs, max_vocab=max_vocab, path='.')

	config = configure("transformer", config)

	metrics = [accuracy]
	if f1: 
		f1 = FBeta()
		metrics.append(f1)
	model = init_model(tdb, Transformer, model_name="transformer", metrics=metrics, config=config, summary=summary)
	
	cbs = []
	if save: cbs.append(SaveModelCallback(clf, every='epoch', monitor='accuracy', name='transformer'))
	if tb is not None: cbs.append(LearnerTensorboardWriter(model, base_dir=Path(tb), name="run"))
	if fp16: model = model.to_fp16()
	model = fit_model(model, epochs, lr, callbacks=cbs)

	# eval, interpret