
import psutil
try: import humanize, GPUtil, wget
except: pass
import os
import torch
import sys
from types import ModuleType, FunctionType
from gc import get_referents
try: from google.colab import drive
except: pass
import shutil
import numpy as np
from tokenizer import SpecialTokenizer, _words_tokenizer
from fastai.text.models.transformer import init_transformer
from fastai import *
from fastai.text import *
from fastai.callbacks import *
import pandas as pd
import numpy as np
import gc

GeLU = Activation.GeLU

BLACKLIST = type, ModuleType, FunctionType

# add default for each model
_default_config = {"awd_lstm": dict(emb_sz=64*4, n_hid=32, n_layers=4, 
									pad_token=0, qrnn=False, bidir=False, 
									output_p=0.4, hidden_p=0.3, input_p=0.4, 
									embed_p=0.05, weight_p=0.5),
					"transformer": dict(ctx_len=512, n_layers=2, n_heads=16, d_model=32, 
										d_head=2, d_inner=64, resid_p=0.2, attn_p=0.2, ff_p=0.2, 
										embed_p=0.2, output_p=0., bias=True, scale=True, act=GeLU, 
										double_drop=False,init=init_transformer, mask=False),
					"seqtolang": dict(num_of_ngrams=159694, output_size=21, 
                 					  hidden_size=64, embedding_dim=64, drp=0.2, 
                 					  n_rounds=2, n_layers=2, bidirectional=True, 
                   					  batch_first=True)
					}

def get_gpu(multiple=False):
	gpus = GPUtil.getGPUs()
	if multiple and len(gpus)>1:
		return gpus, len(gpus)
	else: 
		return gpus[0]

def empty_cache():
	torch.cuda.empty_cache()

def memory():
	process = psutil.Process(os.getpid())
	print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + 
			humanize.naturalsize( process.memory_info().rss))
	print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, 
																								gpu.memoryUsed, 
																								gpu.memoryUtil*100, 
																								gpu.memoryTotal))


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size



def download_url(name="europarl", url="http://www.statmt.org/europarl/v7/europarl.tgz"):
	print(f'Starting download dataset {name}...')

	wget.download(url, url.split("/")[-1])

	print("download finished.")

def read_ds(trn="/content/drive/My Drive/europarl/train_ds.pkl", 
			val="/content/drive/My Drive/europarl/valid_ds.pkl", 
			tst=None):

	train_ds = pd.read_pickle(trn)
	valid_ds = pd.read_pickle(val)
	test_ds = None if tst is None else pd.read_pickle(tst)
	
	return train_ds, valid_ds, test_ds

def cp(src:str, dst:str):
	shutil.copyfile(src, dst)

def cpr(src_dir:str, dst_dir:str):
    src_dir = Path(src_dir)
    if not Path(dst_dir).exists(): Path(dst_dir).mkdir()
    for f in src_dir.ls():
        if not f.is_dir():
            f = str(f).split("\\")[-1]
            cp(str(src_dir)+f"/{f}", dst_dir+f"/{f}")

def mount_drive():
	drive.mount('/content/drive')

def get_tokens(t):
    return list(t.vocab.get_vocab().keys())

def flatten(rank_2):
    return list(itertools.chain.from_iterable(rank_2))

def sup_two(row):
    if len(row['text'].split()) == 1: return 0
    else: return row['text']

def sample(n, train_df, valid_df, p=0.7, seed=None):
	if seed is not None: np.random.seed(seed)
	t_choices = np.random.choice(len(train_df), int(n*p))
	v_choices = np.random.choice(len(valid_df), int(n*(1-p)))

	train_df = train_df.loc[t_choices]
	valid_df = valid_df.loc[v_choices]

	train_df['text'] = train_df.apply(sup_two, axis=1)
	valid_df['text'] = valid_df.apply(sup_two, axis=1)

	train_df = train_df[train_df.text != False]
	valid_df = valid_df[valid_df.text != False]

	train_df.reset_index(drop=True, inplace=True)
	valid_df.reset_index(drop=True, inplace=True)

	return train_df, valid_df

def max_min(df):
	max_len = max([len(x.split()) for x in df.text])
	min_len = min([len(x.split()) for x in df.text])
	return max_len, min_len

def tokenize(t, texts):
	return [flatten(t.tokenize_single(x)) for x in texts]

def configure(model, config=None):
	if config is None: config = _default_config[model]
	else:
		if os.path.isfile(config):
			with open(config, encoding="utf-8") as f:
				config = json.load(f)
		else: config = _default_config[model]
	return config

def init_databunch(trn_df, val_df, t, bs=64, max_vocab=200000, path='.'):

	v = Vocab(get_tokens(t))

	trn_tok = tokenize(t, trn_df["text"])
	val_tok = tokenize(t, val_df["text"])

	trn_lbls = trn_df.label
	val_lbls = val_df.label
	print(len(trn_tok))
	tdb = TextClasDataBunch.from_tokens(path, trn_tok, trn_lbls, val_tok, val_lbls, vocab=v, max_vocab=max_vocab, bs=bs)
	gc.collect()
	return tdb

def init_model(tdb, arch, model_name, metrics=None, config=None, summary=False):
	# create model
	if metrics == None: metrics = [accuracy]
	m = text_classifier_learner(tdb, arch, drop_mult=0.5, metrics=metrics, 
                                pretrained=False, config=config)
	if summary: print(m.summary())
	return m

def lr_find(model):
	model.lr_find()
	model.recorder.plot()

def fit_model(model, epochs, lr, callbacks=[], one_cycle=True):
	if one_cycle:
		if len(callbacks) >= 1:
			model.fit_one_cycle(epochs, lr, callbacks=callbacks)
		else:
			model.fit_one_cycle(epochs, lr)
	else:
		if len(callbacks) >= 1:
			model.fit(epochs, lr, callbacks=callbacks)
		else:
			model.fit(epochs, lr)
	return model

def show_graph(model, grid=True):
	model.recorder.plot_losses(show_grid=grid)

