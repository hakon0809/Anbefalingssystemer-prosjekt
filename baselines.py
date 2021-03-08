import numpy as np
from datetime import datetime

class MostPopularRecommender:
    decayFactor = 604800.0 / 900.0 # number of seconds in a week, divided into 900 time segments (~11 minutes)
        # this makes it take 1 week for an article to get 0.0001 potency (0.99^900 ~ 0.0001)

    def __init__(self, articles):
        self.articles = articles

    def predictNextClick(self, userID, time, k=10):
        # get only articles that have been published (maybe redundant)
        return (self.articles.query(f"(publishtime <= {time})")
                    .assign(score = lambda a: a["totalActiveTime"] * pow(0.99,((time - a["publishtime"]) / self.decayFactor))) # score is active time, with exponential decay over time
                    .sort_values(by="score", ascending=False) # sort by score, highest first
                    .head(k) # take top 10 or k articles
                    .get("documentId") # get only their documentId
                    .to_list()) # return them as a list
        

class MostRecentRecommender:
    def __init__(self, articles):
        self.articles = articles

    def predictNextClick(self, userId, time, k=10):
        pastArticles = self.articles[self.articles['publishtime'] <= time]
        return (pastArticles
                    .sort_values(by="publishtime", ascending=True)
                    .head(k)
                    .get("publishtime")
                    .to_list())

class MeanScoreRecommender:
    def __init__(self, articles, users):
        self.articles = articles
        self.users = users

    def predictNextClick(self, userId, time, k=10):
        # TODO
        return self.articles.query(f"publishtime <= {time}")
        # user mean active time + article mean active time / 2