from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from preprocess_tweets import samples, labels, class_names

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim = len(word2vec)
            print("dim", self.dim)    
        else:
            self.dim=0
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec)
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


X , y = [], []
for i,sample in enumerate(samples):
    X.append(sample.split(" "))
    y.append(class_names[labels[i]-1])

X, y = np.array(X), np.array(y)
print ("total examples %s" % len(y))
GLOVE_27B_50D_PATH = "../datasets/glove.twitter.27B.50d.txt"
GLOVE_27B_200D_PATH = "../datasets/glove.twitter.27B.200d.txt"
glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_27B_50D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode("utf-8")
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums

glove_big = {}
with open(GLOVE_27B_200D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode("utf-8")
        if word in all_words:
            nums=np.array(parts[1:], dtype=np.float32)
            glove_big[word] = nums

model = Word2Vec(X, vector_size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}

print(len(all_words))

# start with the classics - naive bayes of the multinomial and bernoulli varieties
# with either pure counts or tfidf features
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_glove_big = Pipeline([("glove vectorizer",  MeanEmbeddingVectorizer(glove_big)), ("linear svc", SVC(kernel="linear"))])
svc_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), ("linear svc", SVC(kernel="linear"))])

# Extra Trees classifier is almost universally great, let's stack it with our embeddings
etree_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])




# all_models = [
#     ("mult_nb", mult_nb),
#     ("mult_nb_tfidf", mult_nb_tfidf),
#     ("bern_nb", bern_nb),
#     ("bern_nb_tfidf", bern_nb_tfidf),
#     ("svc", svc),
#     ("svc_tfidf", svc_tfidf),
#     ("w2v", etree_w2v),
#     ("w2v_tfidf", etree_w2v_tfidf),
#     ("glove_small", etree_glove_small),
#     ("glove_small_tfidf", etree_glove_small_tfidf),
#     ("glove_big", etree_glove_big),
#     ("glove_big_tfidf", etree_glove_big_tfidf),
#     ("svc_glove_big",svc_glove_big),
#     ("svc_tfidf_glove_big",svc_tfidf_glove_big),
#     ("svc_glove_small",svc_glove_small),
#     ("svc_tfidf_glove_small",svc_tfidf_glove_small)
# ]

all_models = [
    ("Multinomial Naive Bayes", "Count Vectorizer (Non Glove)", mult_nb),
    ("Multinomial Naive Bayes", "Tfidf Vectorizer (Non Glove)", mult_nb_tfidf),
    ("Support Vector Machine", "Count Vectorizer (Non Glove)", svc),
    ("Support Vector Machine", "Tfidf Vectorizer (Non Glove)", svc_tfidf),
    ("Support Vector Machine", "Mean Embedding Vectorizer (Glove)", svc_glove_big),
    ("Support Vector Machine", "Tfidf Embedding Vectorizer (Glove)", svc_glove_big_tfidf),
    ("Extra Trees Classifier", "Mean Embedding Vectorizer (Glove)", etree_glove_big),
    ("Extra Trees Classifier", "Tfidf Embedding Vectorizer (Glove)", etree_glove_big_tfidf),
]

scoring = ['precision_macro', 'recall_macro', 'f1_macro']
unsorted_scores = []
for name, vectorizer, model in all_models:
    accuracy = cross_val_score(model, X, y, cv=5).mean()
    scores = cross_validate(model, X, y, scoring=scoring)
    precision =  scores["test_precision_macro"].mean()
    recall = scores["test_recall_macro"].mean()
    f1 = scores["test_f1_macro"].mean()
    unsorted_scores.append((name, vectorizer, accuracy, precision, recall, f1))
# unsorted_scores = [(name, vectorizer,[ cross_validate(model, X, y, cv=5)]) for name, vectorizer, model in all_models]
# # unsorted_scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in all_models]
sorted_scores = sorted(unsorted_scores, key=lambda x: -x[2])
headers=("model", "vectorizer", "accuracy", "precision", "recall", "f1 score")
print (tabulate(sorted_scores, floatfmt=".4f", headers=headers))