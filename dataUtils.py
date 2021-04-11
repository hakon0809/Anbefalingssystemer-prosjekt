import json
import os
import pandas as pd
import datetime as dt
import time

class DataUtils:
    forbidden_urls = ["http://adressa.no", "http://adressa.no/vaere", "https://adressa.no", "https://adressa.no/vaere"]

    def __init__(self):
        self.knownUserIDs = {}
        self.nextUserID = 0
        self.knownDocumentIDs = {}
        self.nextDocumentID = 0
        #self.knownEventIDs = {}
        #self.nextEventID = 0

    def load_data(self, path, num=0):
        data=[]
        for file in os.listdir(path):
            filename = os.path.join(path, file)

            if os.path.isfile(filename):
                for line in open(filename):
                    entry = json.loads(line.strip())

                    if not entry is None:
                        data.append(entry)
            num -= 1
            if num == 0:
                break # breaking only loads the first num files, 0 means all
        return data

    def filter_data(self, entries):
        # Throw out any entries (events) which are for blacklisted urls (uninteresting data)
        # Throw out any entries which lack a documentId (for now, can be improved later)
        # TODO: try to not throw out missing documentID, only forbidden_urls
        return [e for e in entries if e["url"] not in self.forbidden_urls and e["documentId"] is not None]

    def index_data(self, entries):
        indexed = []
        for i, entry in enumerate(entries):
            indexedEntry = entry.copy()
            indexedEntry["eventId"] = i
            #self.knownEventIDs[entry["eventId"]] = i
            #self.nextEventID = i + 1

            # Get UUID, find its sequential user ID (create new if not found)
            originalUserId = entry["userId"]
            if originalUserId in self.knownUserIDs:
                indexedEntry["userId"]  = self.knownUserIDs[originalUserId]
            else:
                indexedEntry["userId"] = self.nextUserID
                self.knownUserIDs[originalUserId] = self.nextUserID
                self.nextUserID += 1

            # Get document ID, find its sequential ID
            originalDocID = entry["documentId"]
            if originalDocID in self.knownDocumentIDs:
                indexedEntry["documentId"] = self.knownDocumentIDs[originalDocID]
            else:
                indexedEntry["documentId"] = self.nextDocumentID
                self.knownDocumentIDs[originalDocID] = self.nextDocumentID
                self.nextDocumentID += 1
            
            indexed.append(indexedEntry)
        return indexed
    
    def get_dataframe(self, entries):
        #print(entries[0])
        return pd.DataFrame(entries)

    def process_data(self, events):

        # Convert category metadata from "a|b" to ["a","b"]
        events = events.assign(categories=lambda e: e["category"].str.split("|"))
        del events['category']

        # Make list of all unique categories
        allCategories = set()
        for categories in events["categories"]:
            if categories is not None: 
                allCategories.update(categories)

        # Convert time from unix epoch time to python datetime (or, dont do that)
        events['eventTime'] = events['time'] #.apply(lambda x: pd.to_datetime(x, unit="s", utc=True))
        del events['time']

        # Convert publish time from string unix time
        events['publishtime'] = events['publishtime'].apply(lambda x: pd.to_datetime(x).to_numpy().astype('datetime64[s]').astype('int') if x is not None else 0)
        #2017-01-01T20:55:40.000Z

        # TODO: separate out date, time of day, day of the week for easy analysis

        # TODO: If missing date or category metadata, recover from URL (more powerful if we dont filter out missing documentIDs)
        # TODO: distinguish between time of day missing (date recovered from URL) and midnight time 00:00:00
        return events, allCategories