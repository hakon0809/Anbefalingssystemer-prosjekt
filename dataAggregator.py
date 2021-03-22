import pandas as pd

class DataAggregator:

    def __init__(self):
        self.articles = []
    
    def generateArticleData(self, events, nextDocumentID):
        articleList = []
        for id in range(nextDocumentID):
            data = {}
            articleEvents = events.query(f"documentId == {id}")
            data["documentId"] = id
            data["publishtime"] = articleEvents["publishtime"].mode().item()
            categoryMode = articleEvents["categories"].mode()
            data["categories"] = categoryMode.item() if not categoryMode.empty else None
            data["events"] = articleEvents["eventId"].count()
            data["activeEvents"] = articleEvents["activeTime"].count()
            data["totalActiveTime"] = articleEvents["activeTime"].sum(skipna=True)
            articleList.append(data)
        df = pd.DataFrame(articleList)
        df["meanTime"] = df["totalActiveTime"] / df["activeEvents"]
        #df["publishtime"] = df["publishtime"].apply(lambda x: pd.to_datetime(x, unit="s", utc=True))
        self.articles = df