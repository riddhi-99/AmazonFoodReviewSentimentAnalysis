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