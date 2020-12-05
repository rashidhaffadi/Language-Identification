# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""
from text import read_dataframe, sample
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import pickle
# from sklearn.model_selection import train_test_split
import time

# %% Global Variables
languages = ['Bulgarian', 'Czech', 'Danish', 'German', 'Greek', 'English', 'Spanish', 'Estonian', 
             'Finnish', 'French', 'Hungarian', 'Italian', 'Lithuanian', 'Latvian', 'Dutch', 
             'Polish', 'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Swedish']

labels = ['bg','cs','da','de','el','en','es','et', 
            'fi','fr','hu','it','lt','lv','nl','pl',
            'pt','ro','sk','sl','sv']
DEBUG = False
# %%

def _log_reg(nrange, atype):
    return [('vect', CountVectorizer(ngram_range=nrange, analyzer=atype)), 
           ('tfidf', TfidfTransformer(use_idf=False)), 
           ('lrg', LogisticRegression(n_jobs=-1, max_iter=250))]

def _naive_bayes(nrange, atype):
    return [('vect', CountVectorizer(ngram_range=nrange, analyzer=atype)), 
           ('tfidf', TfidfTransformer(use_idf=False)), 
           ('mnn', MultinomialNB())]

def _svm(nrange, atype):
    return [('vect', CountVectorizer(ngram_range=nrange, analyzer=atype)), 
           ('tfidf', TfidfTransformer(use_idf=False)), 
           ('svm', SVC(tol=1e-4))]

def _random_forest(nrange, atype):
    return [('vect', CountVectorizer(ngram_range=nrange, analyzer=atype)), 
           ('tfidf', TfidfTransformer(use_idf=False)), 
           ('rf', RandomForestClassifier(n_estimators=50, min_samples_leaf=3, n_jobs=-1))]
    

def create_pipeline(nrange, atype, m='lg'):
    if m == "svm":
        pip = _svm(nrange, atype)
    elif m == "rf":
        pip = _random_forest(nrange, atype)
    elif m == "nb":
        pip = _naive_bayes(nrange, atype)
    else:
        pip = _log_reg(nrange, atype)
    return Pipeline(pip)

def fit(clf, x_train, y_train):
    
    return clf.fit(x_train, y_train)

def predict(model, x_test):
    return model.predict(x_test)

def assemble(x, y):
    ys = set(y)
    texts = {k:"" for k in ys}
    for x0, y0 in zip(x, y):
        x0 = " " + x0
        texts[y0] += x0
    xs = texts.values()
    return pd.DataFrame(xs), pd.Series(list(ys))

def _model_name(m):    
    if m == "svm": return "(Suport Vector Machine)"
    if m == "rf": return "(Random Forest Classifier)"
    if m == "nb": return "(Multinomial Naive Bayes)"
    else: return "(Logistic Regression)"

def main(param, x_train, x_test, y_train, y_test, languages=languages, m="lg"):
    scores = pd.DataFrame(languages, columns=['languages'])

    for atype in param.keys():
        ngrams = param[atype]
        for n in ngrams:
            clf = create_pipeline(n, atype, m)
            if DEBUG: print("training model {} with ngram_range={}, and analyser_type={}.".format(_model_name(m), n, atype))
            model = fit(clf, x_train, y_train)
            y_pred = predict(model, x_test)
            scores['accuracy-score_'+str(n[1])+ '_' +atype] = accuracy_score(y_pred, y_test) # 
            scores['f1-score_'+str(n[1])+ '_' +atype] = f1_score(y_pred, y_test, average=None)
            with open("./models/{}_{}_{}".format(m, atype, n), "wb") as f:
                pickle.dump(clf, f)
    return scores

# def main(config:Param("", str)="./config.json"):
#     # load config {model, languages, params, }
#     config = json.load(config)


if __name__ == "__main__":
    begin = time.time()
    p = 0.25
    train = read_dataframe("train")
    test = read_dataframe("test")
    # test_params = {"char":[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)], 
    #                "word":[(1, 1), (1, 2), (1, 3)]}
    
    test_params = {"char":[(1, 1)]}
    x_train, y_train = train['text'], train['language']
    x_test, y_test = test['text'], test['language']
    scores = main(test_params, x_train, x_test, y_train, y_test, m="nb")
    # save_dataframe("scores", scores)
    # scores = read_dataframe("scores")
    print(scores.head())
    plt.style.use("ggplot")
    scores.mean().plot(figsize=(10, 10), kind='bar')
    print("means are {}".format(scores.mean()))
    end = time.time()
    print("ended in: {}s".format(round(end-begin, 2)))
    
# model * test_lenght * ngrams_range * analyzer_type

# detect first best ngrame_range, then analyzer type

