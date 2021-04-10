import numpy as np
import pandas as pd
from datetime import datetime

class MostPopularRecommender:
    decayFactor = 604800.0 / 900.0 # number of seconds in a week, divided into 900 time segments (~11 minutes)
        # this makes it take 1 week for an article to get 0.0001 potency (0.99^900 ~ 0.0001)

    def __init__(self, articles, splitEventId):
        self.articles = articles
        # Estimate a retroactive score (biased towards recency over events, but corrects itself quickly)
        self.scores = articles.reset_index().assign(score = lambda a: a["totalActiveTime"] * pow(0.99, splitEventId - a["firstEventId"]))[["documentId", "score"]].set_index("documentId")
    
    def add_event(self, event):
        self.scores["score"] = self.scores["score"] * 0.99
        activeTime = event["activeTime"]
        if np.isnan(activeTime):
            activeTime = self.articles["averageActiveTime"].mean()
        try:
            self.scores.at[event["documentId"], "score"] += activeTime
        except KeyError:
            # document referred to in event isn't scored yet
            self.scores.loc[event["documentId"]] = {"score": activeTime}

    def predictNextClick(self, userID, time, k=10):
        return (self.scores.reset_index().sort_values(by="score", ascending=False).head(k).get("documentId").to_list())

        # get only articles that have been published (maybe redundant)
        '''
        return (self.articles.query(f"(publishtime <= {time})")
                    .assign(score = lambda a: a["totalActiveTime"] * pow(0.99,((time - a["publishtime"]) / self.decayFactor))) # score is active time, with exponential decay over time
                    .sort_values(by="score", ascending=False) # sort by score, highest first
                    .head(k) # take top 10 or k articles
                    .index # get only their documentId (which is the index column)
                    .to_list()) # return them as a list
        '''

class MostRecentRecommender:
    def __init__(self, articles):
        self.articles = articles

    def predictNextClick(self, userId, time, k=10):
        return (self.articles.reset_index()
                    .query(f"(firstEventTime <= {time})")
                    .sort_values(by="firstEventTime", ascending=False)
                    .head(k)
                    .get("documentId")
                    .to_list())

class MeanScoreRecommender:
    def __init__(self, articles, users):
        self.articles = articles
        self.users = users
    
    def predictScore(self, userId, articleId, time):
        cum = 0
        cnt = 0 
        try:
            userMeanViewTime = self.users.at[userId, "averageViewTime"] #select by id and get avg viewing time
            if not np.isnan(userMeanViewTime):
                cum += userMeanViewTime
                cnt += 1
        except KeyError:
            # If the user hasn't been seen before (id not in users dataframe), then they haven't viewed any articles before
            pass
        try:
            articleMeanActiveTime = self.articles.at[articleId, "averageActiveTime"]
            if not np.isnan(articleMeanActiveTime):
                cum += articleMeanActiveTime
                cnt += 1
        except KeyError:
            pass
        return cum / cnt if cnt > 0 else self.articles["averageActiveTime"].mean()
