from Utility.reviewsentimentscore import ReviewSentimentScore
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
analyzer = SentimentIntensityAnalyzer()
text = ''
with open("filter-data.json", "r") as f:
    text = f.read()   
ListReview = json.loads(text)
ListSentimentScore = []
insight_dict = {} ##
for review in ListReview:
    rating = review["score"]
    year = review["year"]
    Score = analyzer.polarity_scores(review["summary"])
    summaryScore = Score["compound"]
    Score = analyzer.polarity_scores(review["text"])
    textScore = Score["compound"]
    ListSentimentScore.append(ReviewSentimentScore(rating,summaryScore,textScore,year))

##############
distinct_year = set([review.year for review in ListSentimentScore])
for dyear in distinct_year:
    s_score = [review.summaryScore for review in ListSentimentScore if review.year == dyear]
    s_score_positive = [review.summaryScore for review in ListSentimentScore if review.year == dyear and review.summaryScore >0.5]
    s_score_negetive = [review.summaryScore for review in ListSentimentScore if review.year == dyear and review.summaryScore < -0.5]
    s_score_nutral = [review.summaryScore for review in ListSentimentScore if review.year == dyear and review.summaryScore > -0.5 and review.summaryScore < 0.5]
    s_score_avg = sum(s_score)/len(s_score)
    s_score_max = max(s_score)
    s_score_min = min(s_score)

    t_score = [review.textScore for review in ListSentimentScore if review.year == dyear]
    t_score_positive = [review.textScore for review in ListSentimentScore if review.year == dyear and review.textScore >0.5]
    t_score_negetive = [review.textScore for review in ListSentimentScore if review.year == dyear and review.textScore < -0.5]
    t_score_nutral = [review.textScore for review in ListSentimentScore if review.year == dyear and review.textScore > -0.5 and review.textScore < 0.5]
    t_score_avg = sum(t_score)/len(t_score)
    t_score_max = max(t_score)
    t_score_min = min(t_score)

    ratings = [review.rating for review in ListSentimentScore if review.year == dyear]
    ratings_dict = {}
    for r in ratings:
        if r in ratings_dict:
            ratings_dict[r] += 1
        else:
            ratings_dict[r] = 1

    insight_dict[dyear] = {"ratings": ratings_dict,
                           "summary_score_positive_count" : len(s_score_positive),
                           "summary_score_negetive_count" : len(s_score_negetive),
                           "summary_score_nutral_count" : len(s_score_nutral),
                           "summary_score_max": s_score_max,
                            "summary_score_min": s_score_min,
                            "summary_score_avg": s_score_avg,
                            "text_score_positive_count" : len(t_score_positive),
                            "text_score_negetive_count" : len(t_score_negetive),
                            "text_score_nutral_count" : len(t_score_nutral),
                            "text_score_max": t_score_max,
                            "text_score_min": t_score_min,
                            "text_score_avg": t_score_avg}

a_dict = {key: insight_dict[key] for key in insight_dict if key != '0'}
print(a_dict)
#Create Json File sentiment score
with open("sentiment-score.json", "w") as outfile:
    json.dump(a_dict, outfile)