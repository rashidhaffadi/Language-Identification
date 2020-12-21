# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from fastcore.script import *
from tqdm import tqdm


plt.style.use('ggplot')

def x_(dim, scale=3):
    x = scale * np.random.rand(dim)
    return np.sort(x)
    
def ε_(dim, r=0.1):
    return r * np.random.randn(dim)

def y_(x):
    return np.sin((0.5*np.pi) * x)

def f(x, true=False):
    dim = len(x)
    if not true: ε = ε_(dim)
    y = y_(x)
    return y if true else y + ε

def fit(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

def transform(pl, x):
    x = x.reshape((-1, 1))
    return pl.fit_transform(x)

def error(model, x, y_true):
    y_pred = model.predict(x)
    return mean_squared_error(y_pred, y_true)

def true_curve(x):
    return x, f(x, True)

def model_curve(x, model, pl, scale):
    return x, model.predict(x)

def show(trn_x, trn_y, tst_x, tst_y, x0, y_true, y_preds, d, scale):
    plt.plot(trn_x, trn_y, 'ko', label = 'Train Observations')
    plt.plot(tst_x, tst_y, 'r*', label = 'Test Observations')
    plt.plot(x0, y_true, linewidth = 4, label = 'True Function')
    plt.plot(x0, y_preds, linewidth = 4, label = 'Model Function')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend()
    plt.ylim(-1.5, 1.5); plt.xlim(0, scale)
    # plt.title('{} Degree Model on Training Data'.format(d))
    os.makedirs("./img/over_under", exist_ok=True)
    plt.savefig("./img/over_under/{}.png".format(d))
    plt.close()
    # plt.show()


def train(trn_x, trn_y, tst_x, tst_y, d=1, scale=2):
    if d < 1: return 
    pl = PolynomialFeatures(degree=d, include_bias=False)
    trn_x = transform(pl, trn_x)
    # print(trn_x.shape)
    model = fit(trn_x, trn_y)
    x0 = np.linspace(0, scale, 1000)
    _, y_true = true_curve(x0)
    x = transform(pl, x0)
    _, y_preds = model_curve(x, model, pl, scale)
    show(trn_x[:, 0], trn_y, tst_x, tst_y, x0, y_true, y_preds, d, scale)
    tst_x = transform(pl, tst_x)
    return round(error(model, trn_x, trn_y), 2), round(error(model, tst_x, tst_y), 2)
    
    
@call_parse
def main(dim:Param("", int)=200,
         p:Param("", float)=0.75,
         scale:Param("", int)=3,
         drange:Param("", int)=30,
         step:Param("", int)=2):

    x = x_(dim, scale)
    y = f(x)

    trn_x, trn_y = x[:int(p * dim)], y[:int(p * dim)]
    tst_x, tst_y = x[int(p * dim):], y[int(p * dim):]

    trn_errs, tst_errs, ds = [], [], [i for i in range(1, drange, step)]
    for d in tqdm(ds):
        trn_err, tst_err = train(trn_x, trn_y, tst_x, tst_y, d, scale=scale)
        tst_errs.append(tst_err)
        trn_errs.append(trn_err)

    res = pd.DataFrame({"dimension":ds, "train-error":trn_errs, "test-error":tst_errs, 
                        "difference":[ts-tr for tr, ts in zip(trn_errs, tst_errs)]})
        
    print('errors: ')
    print(res)

# if __name__ == "__main__":
#     dim, p, scale = 200, 0.7, 3

#     x = x_(dim, scale)
#     y = f(x)

#     trn_x, trn_y = x[:int(p * dim)], y[:int(p * dim)]
#     tst_x, tst_y = x[int(p * dim):], y[int(p * dim):]
#     errors = []
#     for d in [1, 2, 3, 4, 5, 25, 26, 27]:
#         errors.append(main(trn_x, trn_y, tst_x, tst_y, d, scale=scale))
        
#     print(errors)

