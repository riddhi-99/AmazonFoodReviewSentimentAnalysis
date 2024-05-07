import warnings
warnings.filterwarnings("ignore")

# Data Loading and other standard packages
import sqlite3
import nltk
import string
import re
import pickle
import os
import time
import string
import pickle
import pandas as pd
import numpy as np
from scipy import interp
from tqdm import tqdm
from scipy.sparse import find
from wordcloud import WordCloud, STOPWORDS

# Plotting Packages
import matplotlib.pyplot as plt

# Preprocessing packages
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

# Packages related to text processing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import decomposition
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# Packages for performance metrics
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Packages for crossvalidation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold

# Model packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from bs4 import BeautifulSoup

from Utility.commonclass import CommonClass

class LRInference:
  def __init__(self, vect_pickle, std_pickle, model_pickle ,y_train_pickle):
    preprocessed_reviews_train = CommonClass.pickle_load(vect_pickle) #"preprocessed_reviews_train.p"
    self.count_vect = CountVectorizer()
    self.count_vect.fit(preprocessed_reviews_train)
    bow_features = self.count_vect.get_feature_names_out()

    y_train = CommonClass.pickle_load(y_train_pickle)#"y_train.p"
    lr_model = CommonClass.pickle_load(model_pickle) #"logistic_reg_model.p"
    final_counts_train_std = CommonClass.pickle_load(std_pickle) #"final_counts_train_std.p"

    self.lr_optimal = LogisticRegression(penalty=lr_model.best_params_['penalty'],C=lr_model.best_params_['C'])
    self.lr_optimal.fit(final_counts_train_std, y_train)

  def perform_standardization(self, text_vect):
    if (type(text_vect).__name__ == 'csr_matrix'):
        scaler = MaxAbsScaler().fit(text_vect)
        x_train_std = scaler.transform(text_vect)
        return x_train_std
    else:
        scaler = StandardScaler().fit(text_vect)
        x_train_std = scaler.transform(text_vect)
        return x_train_std

  def model_pred(self, text):
    text_vect = self.count_vect.transform([text])
    text_vect_std = self.perform_standardization(text_vect)
    log_proba = self.lr_optimal.predict_log_proba(text_vect_std)
    if (1 / (1 + np.exp(-log_proba)))[0].argmax() == 1:
      return "Positive", (1 / (1 + np.exp(-log_proba)))[0].max()
    else:
      return "Negative", (1 / (1 + np.exp(-log_proba)))[0].max()
    # return log_proba
