import pandas as pd

class DataAggregator:

    def __init__(self):
        self.articles = None
        self.users = None
    
    def generateArticleData(self, events, nextDocumentID):
        articleList = []
        for id in range(nextDocumentID):
            data = {}
            articleEvents = events.query(f"documentId == {id}")
            data["documentId"] = id
            data["publishtime"] = articleEvents["publishtime"].mode().item() # looks if any event has publishtime
            categoryMode = articleEvents["categories"].mode() # looks if any event has categories (mode is most common value except None)
            data["categories"] = categoryMode.item() if not categoryMode.empty else None # (if no event has category, save it as None)
            data["events"] = articleEvents["eventId"].count() # number of events involving this article
            data["activeEvents"] = articleEvents["activeTime"].count() # number of events with registered active time
            data["totalActiveTime"] = articleEvents["activeTime"].sum(skipna=True) # total active time registered on this article
            articleList.append(data)
        df = pd.DataFrame(articleList) # turn list of dicts into dataframe
        df["meanTime"] = df["totalActiveTime"] / df["activeEvents"] # calculate the average active time (excluding events with no active time)
        self.articles = df

    def generateUserData(self, events, nextUserID):
        userList = []
        for id in range(nextUserID):
            data = {}
            userEvents = events.query(f"userId == {id}")
            categoriesSlice = userEvents["categories"].dropna()
            data["userId"] = id
            data["articlesViewed"] = len(userEvents.index) # total number of articles this user has looked at
            data["averageViewTime"] = userEvents["activeTime"].mean() # average view time across all of the user's articles
            data["totalViewTime"] = userEvents["activeTime"].sum()
            #data["averageArticlePublishTime"] = # TODO: get 24h time (should be precalculated in dataUtils)
            #data["averageArticlePublishDay"] = # TODO: get day of the week (should be precalculated in dataUtils)
            handledCategories = set() # TODO: get all categories ahead of time from dataUtils or generateArticleData
            maxCount = 0
            for categoryArr in categoriesSlice:
                for category in categoryArr:
                    if category in handledCategories:
                        continue # skip re-computing categories we've already looked at, for efficiency
                    categoryFilter = categoriesSlice.apply(lambda array: category in array) # filter for all arrays with this category in it 
                    categoryCount = len(categoriesSlice[categoryFilter].index) # count number of rows that pass the filter
                    data[f"{category}Viewed"] = categoryCount # save it in a column for this category specifically
                    if categoryCount > maxCount:
                        data["mostCommonCategory"] = category
                        maxCount = categoryCount
                    handledCategories.add(category)
            userList.append(data)
        df = pd.DataFrame(userList) # turn list of dicts into dataframe
        self.users = df