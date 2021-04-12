import pandas as pd
import numpy as np

class DataAggregator:

    def __init__(self, categories, strict=False):
        self.categories = categories
        self.articles = None
        self.users = None
        self.ratingCol = 'activeTime' if strict else 'filledActiveTime' # use only true values or use normalized values in the ratings matrix
        self.ratings = None
    
    def generateArticleData(self, events, nextDocumentID):
        articleList = []
        for id in range(nextDocumentID):
            data = {}
            articleEvents = events.query(f"documentId == {id}")

            # If we encounter an article with no events, it means we've moved past the train set and into articles that haven't been
            # published yet. Break the loop early and avoid trying to process null data.
            if (len(articleEvents.index) == 0):
                break

            data["documentId"] = id
            data["title"] = articleEvents["title"].mode()
            data["publishtime"] = articleEvents["publishtime"].mode().max().item() # looks if any event has publishtime
            data["firstEventId"] = articleEvents["eventId"].min()
            data["firstEventTime"] = articleEvents.loc[data["firstEventId"]]["eventTime"]
            # TODO: "firstEventTime" that overrides publishtime when publishtime is unavailable
            categoryMode = articleEvents["categories"].mode() # looks if any event has categories (mode is most common value except None)
            data["categories"] = categoryMode.item() if not categoryMode.empty else None # (if no event has category, save it as None)
            data["events"] = articleEvents["eventId"].count() # number of events involving this article
            data["activeEvents"] = articleEvents["activeTime"].count() # number of events with registered active time
            data["totalActiveTime"] = articleEvents["activeTime"].sum(skipna=True) # total active time registered on this article
            
            articleList.append(data)
        df = pd.DataFrame(articleList).set_index("documentId") # turn list of dicts into dataframe
        df["averageActiveTime"] = df["totalActiveTime"] / df["activeEvents"] # calculate the average active time for each article (excluding events with no active time)
        self.articles = df

    def generateUserData(self, events, nextUserID):
        userList = []
        for id in range(nextUserID):
            data = {}
            userEvents = events.query(f"userId == {id}")

            # Same as above: if we encounter an article with no events, it means we've moved past the train set and into users who
            # haven't shown up yet. Break the loop early and avoid trying to process null data.
            if (len(userEvents.index) == 0):
                break

            categoriesSlice = userEvents["categories"].dropna()
            data["userId"] = id
            data["articlesViewed"] = userEvents["documentId"].nunique() # number of articles seen by user
            data["activeEvents"] = userEvents["activeTime"].count() # number of events with a registered activetime (seeing an article twice is two visits, not cumulative time)
            data["totalViewTime"] = userEvents["activeTime"].sum(skipna=True) # total active time registered by this user
            #data["averageArticlePublishTime"] = # TODO: get 24h time (should be precalculated in dataUtils)
            #data["averageArticlePublishDay"] = # TODO: get day of the week (should be precalculated in dataUtils)
            maxArticle = (None, 0)
            for category in self.categories:
                categoryFilter = categoriesSlice.apply(lambda array: category in array) # filter for all arrays with this category in it 
                categoryCount = len(categoriesSlice[categoryFilter].index) # count number of rows that pass the filter
                data[f"{category}Viewed"] = categoryCount # save it in a column for this category specifically
                if categoryCount > maxArticle[1]: # if this category is more common than all previous ones
                    maxArticle = (category, categoryCount) # save it
            data["mostCommonCategory"], data["mostCommonCategoryCount"] = maxArticle
            userList.append(data)
        df = pd.DataFrame(userList).set_index("userId") # turn list of dicts into dataframe
        df["averageViewTime"] = df["totalViewTime"] / df["activeEvents"] # average view time across all of the user's articles
        self.users = df

    def generateRatingMatrix(self, events, totalNumUsers, totalNumDocuments):
        # creates a np matrix where ratings[userId, documentId] = users activetime of document
        df = events[['userId','documentId', self.ratingCol]] # filter only necessary columns
        rating_matrix = np.zeros((totalNumUsers,totalNumDocuments)) # initialize empty matrix
        for _, row in df.iterrows():
            prev = rating_matrix[int(row['userId']), int(row['documentId'])]
            rating_matrix[int(row['userId']), int(row['documentId'])] = max(prev, row[self.ratingCol])
            # two clicks is a strong endorsement, but as a basis for predicted scores that assumes a single click, it will often be too long
        self.ratings = rating_matrix
    
    def update_rating(self, event):
        # only used for rating matrices that are updated between events, live_matrix
        if not (np.isnan(event[self.ratingCol]) or event[self.ratingCol] == 0.0): 
            prev = self.ratings[event["userId"], event["documentId"]]
            self.ratings[event["userId"], event["documentId"]] = max(prev, event[self.ratingCol])

    def add_event(self, event):
        articleId = event["documentId"]
        activeTime = event["activeTime"]
        hasActiveTime = not np.isnan(activeTime) and not activeTime == 0

        try:
            # Article already exists - add to its existing data
            self.articles.at[articleId, "events"] += 1
            if hasActiveTime:
                activeCount = self.articles.at[articleId, "activeEvents"]
                self.articles.at[articleId, "activeEvents"] = activeCount + 1 if not np.isnan(activeCount) else 1

                totalTime = self.articles.at[articleId, "totalActiveTime"]
                self.articles.at[articleId, "totalActiveTime"] = totalTime + activeTime if not np.isnan(totalTime) else activeTime
                self.articles.at[articleId, "averageActiveTime"] = self.articles.at[articleId, "totalActiveTime"] / self.articles.at[articleId, "activeEvents"]

        except KeyError:
            # Article doesn't exist yet - add it to the database
            data = {}
            data["documentId"] = articleId
            data["publishtime"] = event["publishtime"] if event["publishtime"] is not None else event["eventTime"] # TODO: do elsewhere
            data["firstEventId"] = event["eventId"]
            data["firstEventTime"] = event["eventTime"]
            data["categories"] = event["categories"]
            data["events"] = 1
            data["activeEvents"] = 1 if hasActiveTime else 0
            data["totalActiveTime"] = activeTime if hasActiveTime else 0
            data["averageActiveTime"] = activeTime if hasActiveTime else 0
            self.articles.loc[articleId] = data

        userId = event["userId"]
        try:
            articlesViewed = self.users.at[userId, "articlesViewed"]
            self.users.at[userId, "articlesViewed"] = articlesViewed + 1 if not np.isnan(articlesViewed) else 1
            if hasActiveTime:
                activeEvents = self.users.at[userId, "activeEvents"]
                self.users.at[userId, "activeEvents"] = activeEvents + 1 if not np.isnan(activeEvents) else 1
                totalTime = self.users.at[userId, "totalViewTime"]
                self.users.at[userId, "totalViewTime"] = totalTime + activeTime if not np.isnan(totalTime) else activeTime
                self.users.at[userId, "averageViewTime"] = self.users.at[userId, "totalViewTime"] / self.users.at[userId, "activeEvents"]
            categories = event["categories"]
            if categories is not None:
                for category in categories:
                    count = self.users.at[userId, f"{category}Viewed"]
                    newcount = count + 1 if not np.isnan(count) else 1
                    self.users.at[userId, f"{category}Viewed"] = newcount
                    if newcount > self.users.at[userId, "mostCommonCategoryCount"]:
                        self.users.at[userId, "mostCommonCategory"] = category
                        self.users.at[userId, "mostCommonCategoryCount"] = newcount
        except KeyError:
            data = {}
            data["index"] = userId
            data["userId"] = userId
            data["articlesViewed"] = 1
            data["activeEvents"] = 1 if hasActiveTime else 0
            data["totalViewTime"] = activeTime if hasActiveTime else 0
            data["averageViewTime"] = activeTime if hasActiveTime else 0
            categories = event["categories"]
            allCategories = self.categories.copy()
            if categories is not None:
                allCategories -= set(categories)
                for category in categories:
                    data[f"{category}Viewed"] = 1
                    # arbitrarily pick the last category as the most common, since we only have this one event to judge from
                    data["mostCommonCategory"] = category
                    data["mostCommonCategoryCount"] = 1
            for category in allCategories:
                data[f"{category}Viewed"] = 0
            ##print(self.users.loc[userId - 1])
            ##print(data)
            self.users.loc[userId] = data
        