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
