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

    def load_data(self, path):
        data=[]
        for file in os.listdir(path):
            filename = os.path.join(path, file)

            if os.path.isfile(filename):
                for line in open(filename):
                    entry = json.loads(line.strip())

                    if not entry is None:
                        data.append(entry)
            break # TODO: this only loads the first day of data
        return data

    def filter_data(self, entries):
        # Throw out any entries (events) which are for blacklisted urls (uninteresting data)
        # Throw out any entries which lack a documentId (for now, can be improved later)
        return [e for e in entries if e["url"] not in self.forbidden_urls and e["documentId"] is not None]

    def index_data(self, entries):
        for entry in entries:
            # Get UUID, replace it with sequential user ID (create new if not found)
            originalUserId = entry["userId"]
            if originalUserId in self.knownUserIDs:
                entry["userId"] = self.knownUserIDs[originalUserId]
            else:
                entry["userId"] = self.nextUserID
                self.knownUserIDs[originalUserId] = self.nextUserID
                self.nextUserID += 1

            # Get document ID, replace it with sequential ID
            originalDocID = entry["documentId"]
            if originalDocID in self.knownDocumentIDs:
                entry["documentId"] = self.knownDocumentIDs[originalDocID]
            else:
                entry["documentId"] = self.nextDocumentID
                self.knownDocumentIDs[originalDocID] = self.nextDocumentID
                self.nextDocumentID += 1
    
    def get_dataframe(self, entries):
        print(entries[0]);
        return pd.DataFrame(entries)

    def process_data(self, events):
        events.info()
        # Convert category metadata from "a|b" to ["a","b"]
        events = events.assign(categories=lambda e: e["category"].str.split("|")) # this needs testing lol YEAH
        del events['category']

        # Convert time from unix epoch time to python datetime (or, dont do that)
        events['eventTime'] = events['time'] #.apply(lambda x: pd.to_datetime(x, unit="s", utc=True))
        del events['time']

        # Convert publish time from string unix time
        events['publishtime'] = events['publishtime'].apply(lambda x: pd.to_datetime(x).to_numpy().astype('datetime64[s]').astype('int') if x is not None else 0)
        #2017-01-01T20:55:40.000Z

        # TODO: separate out date and time of day for easy analysis

        # TODO: If missing date or category metadata, recover from URL (more powerful if we dont filter out missing documentIDs)
        # TODO: distinguish between time of day missing (date recovered from URL) and midnight
        events.info()
        return events