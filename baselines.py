import numpy as np
import pandas as pd
from datetime import datetime

class MostPopularRecommender:
    def __init__(self, articles, splitEventId):
        self.articles = articles
        # Estimate a retroactive score (biased towards recency over events, but corrects itself quickly)
        #self.scores = articles.reset_index().assign(score = lambda a: a["totalActiveTime"] * pow(0.99, splitEventId - a["firstEventId"]))[["documentId", "score"]].set_index("documentId")
        self.scores = articles.reset_index().assign(score = lambda a: a["events"] * pow(0.99, splitEventId - a["firstEventId"]))[["documentId", "score"]].set_index("documentId")
    
    def add_event(self, event):
        self.scores["score"] = self.scores["score"] * 0.99
        activeTime = 1 # event["activeTime"]
        #if np.isnan(activeTime):
        #    activeTime = self.articles["averageActiveTime"].mean()
        try:
            self.scores.at[event["documentId"], "score"] += activeTime
        except KeyError:
            # document referred to in event isn't scored yet
            self.scores.loc[event["documentId"]] = {"score": activeTime}

    def predictNextClick(self, userID, time, k=10):
        return (self.scores.reset_index().sort_values(by="score", ascending=False).head(k).get("documentId").to_list())

class MostRecentRecommender:
    timeProperty = "publishtime" # publishtime or firstEventTime

    def __init__(self, articles, k=10):
        self.articles = articles
        self.k = k
        self.kMostRecent = list(articles.reset_index().nlargest(k, self.timeProperty)[[self.timeProperty,"documentId"]].to_records(index=False))

    def add_event(self, event):
        if event["documentId"] in [id for (time, id) in self.kMostRecent]:
            return # no duplicates
        article = self.articles.loc[event["documentId"]]
        if article[self.timeProperty] > self.kMostRecent[-1][0]:
            self.kMostRecent[-1] = (article[self.timeProperty], event["documentId"]) # replace oldest
            self.kMostRecent.sort(key=lambda e: e[0], reverse=True)

    def predictNextClick(self, userId, time, k=10):
        return [id for (time, id) in self.kMostRecent]

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
