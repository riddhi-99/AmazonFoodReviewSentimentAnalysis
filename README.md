# FoodReviewAnalysis
Data Source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data
The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.

Number of reviews: 568,454
Number of users: 256,059
Number of products: 74,258
Timespan: Oct 1999 - Oct 2012
Number of Attributes/Columns in data: 10
Attribute Information:

Id
ProductId - unique identifier for the product
UserId - unqiue identifier for the user
ProfileName
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
Score - rating between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review
Objective:
Given a review, determine whether the review is positive (rating of 4 or 5) or negative (rating of 1 or 2).


[Q] How to determine if a review is positive or negative?

[Ans] We could use Score/Rating. A rating of 4 or 5 can be cosnidered as a positive review. A rating of 1 or 2 can be considered as negative one. A review of rating 3 is considered nuetral and such reviews are ignored from our analysis. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

# [1]. Reading Data
# [1.1] Loading the data
The dataset is available in two forms

.csv file
JSON file
In order to load the data, We have used these datasets as it is easier to query the data and visualise the data efficiently.


# Package to filter warnings
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


current_working_directory = os.getcwd() 
current_working_directory = current_working_directory + "\\data\\Reviews.csv"

filtered_data = pd.read_csv(current_working_directory, nrows=50000)

def partition(x):
    if x < 3:
        return 0
    return 1

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
print("Number of data points in our data", filtered_data.shape)

#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape

final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
print(final['Score'].value_counts())

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def split_time_based(df, time_col, train_perc):
    if type(df).__name__ != 'DataFrame':
        print("Please provide a dataframe!")
    if type(time_col).__name__ != 'list':
        print("Please porvide the time column as list")
    # Sort the dataframe based on time column
    df.sort_values(by=time_col, inplace=True)
    df_row, df_col = df.shape
    train_row = round(df_row * (train_perc/100))
    test_row = df_row - train_row
    return df.head(train_row), df.tail(test_row)

final_train, final_test = split_time_based(final, ['Time'], 80)
print(final_train.shape)
print(final_test.shape)

from tqdm import tqdm
preprocessed_reviews_train = []
# tqdm is for printing the status bar
for sentance in tqdm(final_train['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews_train.append(sentance.strip())

# Combining all the above stundents
preprocessed_reviews_test = []
# tqdm is for printing the status bar
for sentance in tqdm(final_test['Text'].values):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews_test.append(sentance.strip())

print("Train: ",len(preprocessed_reviews_train))
print("Test: ",len(preprocessed_reviews_test))


y_train = final_train['Score']
y_test = final_test['Score']

y_train = y_train.values
y_test = y_test.values

def pickle_dump(file_name, file_object):
  obj1 = open(file_name,'wb')
  pickle.dump(file_object, obj1)
  obj1.close()

def pickle_load(file_name):
  obj1 = open(file_name,'rb')
  file_object = pickle.load(obj1)
  obj1.close()
  return file_object
pickle_dump("y_train.p", y_train)
pickle_dump("preprocessed_reviews_train.p", preprocessed_reviews_train)

#BoW for Train data
count_vect = CountVectorizer() #in scikit-learn
count_vect.fit(preprocessed_reviews_train)
bow_features = count_vect.get_feature_names_out()
print("some feature names ", count_vect.get_feature_names_out()[:10])
print('='*50)

final_counts_train = count_vect.transform(preprocessed_reviews_train)
print("the type of count vectorizer ",type(final_counts_train))
print("the shape of out text BOW vectorizer ",final_counts_train.get_shape())
print("the number of unique words ", final_counts_train.get_shape()[1])

final_counts_test = count_vect.transform(preprocessed_reviews_test)
print("the type of count vectorizer ",type(final_counts_test))
print("the shape of out text BOW vectorizer ",final_counts_test.get_shape())
print("the number of unique words ", final_counts_test.get_shape()[1])

#defining function to apply Logistic Regression
def apply_logistic_regression(x_train, y_train, C_values, cross_val_number, regularization_param):
    C_vals = C_values
    tscv = TimeSeriesSplit(n_splits=cross_val_number)

    # define Logistic Regression to use in crossvalidation
    parameter = {'C':C_vals, 'penalty':regularization_param}
    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameter, cv=tscv, scoring='roc_auc', n_jobs=1, verbose=1) #Use GridSearchCV for 10 fold cross validation
    clf.fit(x_train, y_train)
    return clf

def apply_logistic_regression_kfold(x_train, y_train, C_values, cross_val_number, regularization_param):
    kf = KFold(n_splits=cross_val_number)
    kf.get_n_splits(x_train)
    local_auc_score_train = []
    global_auc_score_train = []
    local_auc_score_test = []
    global_auc_score_test = []


    for c_v in tqdm(C_values):
        for train_index, test_index in kf.split(x_train):
            lr = LogisticRegression(penalty=regularization_param[0],C=c_v)
            lr.fit(x_train[train_index], y_train[train_index])
            pred_log_proba = lr.predict_log_proba(x_train[train_index])
            pred_log_proba1 = lr.predict_log_proba(x_train[test_index])
            fpr, tpr, _ = roc_curve(y_train[train_index], pred_log_proba[:,1])
            fpr1, tpr1, _ = roc_curve(y_train[test_index], pred_log_proba1[:,1])
            auc_score = auc(fpr, tpr)
            auc_score1 = auc(fpr1, tpr1)
            local_auc_score_train.append(auc_score)
            local_auc_score_test.append(auc_score1)
        global_auc_score_train.append(sum(local_auc_score_train)/len(local_auc_score_train))
        global_auc_score_test.append(sum(local_auc_score_test)/len(local_auc_score_test))
    return global_auc_score_train,global_auc_score_test

# Define function to standardize data
def perform_standardization(x_train, x_test):
    if (type(x_train).__name__ == 'csr_matrix') and (type(x_test).__name__ == 'csr_matrix'):
        scaler = MaxAbsScaler().fit(x_train)
        x_train_std = scaler.transform(x_train)
        x_test_std = scaler.transform(x_test)
        return x_train_std, x_test_std
    else:
        scaler = StandardScaler().fit(x_train)
        x_train_std = scaler.transform(x_train)
        x_test_std = scaler.transform(x_test)
        return x_train_std, x_test_std

C_values = [10**-4,10**-3,10**-2,10**-1,1,10**1,10**2,10**3,10**4]

final_counts_train_std, final_counts_test_std = perform_standardization(final_counts_train, final_counts_test)
# Apply Logistic Regression on Standardized data for L1 Regularization
cross_val_number = 10
regularization_param = ['l2']
clf = apply_logistic_regression(final_counts_train_std, y_train, C_values, cross_val_number, regularization_param)
pickle_dump("logistic_reg_model.p", clf)
pickle_dump("final_counts_train_std.p", final_counts_train_std)
print("Best HyperParameter: ",clf.best_params_)
print("Best Score: %.2f%%"%(clf.best_score_*100))


# [2] Exploratory Data Analysis
# [2.1] Data Cleaning: Deduplication
It is observed (as shown in the table below) that the reviews data had many duplicate entries. Hence it was necessary to remove duplicates in order to get unbiased results for the analysis of the data. Following is an example:
import os
from Utility.review import Review
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import time
import json

analyzer = SentimentIntensityAnalyzer()

current_working_directory = os.getcwd() 
current_working_directory = current_working_directory + "\\data\\foods.txt"
text = ''

for i in tqdm(range (101), 
               desc="Loading…", 
               ascii=False, ncols=75):
    time.sleep(0.01)     


with open(current_working_directory) as f:
    text = f.read()   
print("Processing food data...")
reviews = text.split('\n\n')
ListReview = []
for review in reviews:
    lines = review.split('\n')
    score = 0
    summary = ""
    text = ""
    year = 0
    for line in lines:
        if line.__contains__("review/score:"):
            score = line.replace("review/score:","")
        elif line.__contains__("review/summary:"):
            summary = line.replace("review/summary:","")  
        elif line.__contains__("review/text:"):
            text = line.replace("review/text:","")
        elif line.__contains__("review/time:"):
            time = line.replace("review/time:","")
            year = datetime.fromtimestamp(int(time)).year  

    ListReview.append(Review(score,summary,text,year)) 

review_dict = []
for review in ListReview:
    _dic = {"score": review.score , "summary": review.summary , "text": review.text , "year": review.year }
    review_dict.append(_dic)
with open("filter-data.json", "w") as outfile:
    json.dump(review_dict, outfile)


# [3] Preprocessing
# [3.1]. Preprocessing Review Text
Now that we have finished deduplication our data requires some preprocessing before we go on further with analysis and making the prediction model.

Hence in the Preprocessing phase we do the following in the order below:-

Begin by removing the html tags
Remove any punctuations or limited set of special characters like , or . or # etc.
Check if the word is made up of english letters and is not alpha-numeric
Check to see if the length of the word is greater than 2 (as it was researched that there is no adjective in 2-letters)
Convert the word to lowercase
Remove Stopwords
Finally Snowball Stemming the word (it was obsereved to be better than Porter Stemming)
After which we collect the words used to describe positive and negative reviews

Here as we only want to get the global sentiment of the recommendations (positive or negative), we will purposefully ignore all Scores equal to 3. If the score is above 3, then the recommendation wil be set to "positive". Otherwise, it will be set to "negative".
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

# [4] Featurization
# [4.1] BAG OF WORDS
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def split_time_based(df, time_col, train_perc):
    if type(df).__name__ != 'DataFrame':
        print("Please provide a dataframe!")
    if type(time_col).__name__ != 'list':
        print("Please porvide the time column as list")
    # Sort the dataframe based on time column
    df.sort_values(by=time_col, inplace=True)
    df_row, df_col = df.shape
    train_row = round(df_row * (train_perc/100))
    test_row = df_row - train_row
    return df.head(train_row), df.tail(test_row)

final_train, final_test = split_time_based(final, ['Time'], 80)
print(final_train.shape)
print(final_test.shape)

# [4.2.1] Converting text into vectors using Avg W2V, TFIDF-W2V
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
      return "Negative", (1 / (1 + np.exp(-log_proba)))[0].max(); 
    
  


  # [5] Applying Logistic Regression
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

   # [6] Conclusions

   # Packages
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import json
from Utility.LRInference import LRInference

def barGrapPlot(xAxis, yAxis_s,yAxis_t, title):
    X_axis = np.arange(len(xAxis)) 
    plt.bar(X_axis - 0.2,yAxis_s,0.4,label = "Title")
    plt.bar(X_axis + 0.2,yAxis_t,0.4,label = "Review")
    plt.xticks(X_axis, xAxis) 
    plt.title(title)
    plt.legend()
    plt.show()

print("Complete.")
option = input("Choose Option:\n1.Words based on rating\n2.Insight based on year\n3.Test text sentiment\n")    
if(option == "1"):
    rating = input("Ente rating between 1 to 5:\n")
    stopwords = set(STOPWORDS)
    stopwords.update(["Amazon", "Product", "br"])
    with open("filter-data.json", "r") as f:
        text = f.read()   
    ListReview = json.loads(text)
    all_rating_1 = " ".join([review["text"] for review in ListReview if review["score"]== " "+rating+".0"])
    wordcloud_rating_1 = WordCloud(stopwords=stopwords, background_color="white").generate(all_rating_1)
    plt.imshow(wordcloud_rating_1, interpolation='bilinear')
    plt.axis("off")
    plt.show()
elif(option == "2"):
    year = input("Ente year between 1999 to 2012:\n")
    with open("sentiment-score.json", "r") as f:
        text = f.read()   
    insight_dict = json.loads(text)
    xAxis  = ["Positive","Negetive", "Nutral"]
    yAxis_s = [insight_dict[year]["summary_score_positive_count"],insight_dict[year]["summary_score_negetive_count"],insight_dict[year]["summary_score_nutral_count"]]
    yAxis_t = [insight_dict[year]["text_score_positive_count"],insight_dict[year]["text_score_negetive_count"],insight_dict[year]["text_score_nutral_count"]]
    barGrapPlot(xAxis,yAxis_s,yAxis_t,year)
elif(option == "3"):
    logistic_reg_infer = LRInference("preprocessed_reviews_train.p",
                                 "final_counts_train_std.p",
                                 "logistic_reg_model.p",
                                 "y_train.p")
    text = input("Enter a sentence:\n")
    sentiment, score = logistic_reg_infer.model_pred(text)
    print("Sentiment: ",sentiment)
    print("Score: ",score)
else:
    print("Please choose a correct option.")
    
# output 

![Alt text]( https://github.com/riddhi-99/AmazonFoodReviewSentimentAnalysis/blob/main/Picture1.png)
![Alt text](https://github.com/riddhi-99/AmazonFoodReviewSentimentAnalysis/blob/main/Picture2.png)

![Alt text](https://github.com/riddhi-99/AmazonFoodReviewSentimentAnalysis/blob/main/WhatsApp%20Image%202024-05-08%20at%2018.42.31_8a5080b0.jpg)

# Overall Observation
Features are multi-collinear, so before modelling we need to discard redundant features
TFIDF Featurization performs best with accuracy of 93.512

   
