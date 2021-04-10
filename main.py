import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing

from tabulate import tabulate

from dataUtils import DataUtils
from dataAggregator import DataAggregator
from baselines import MostPopularRecommender, MostRecentRecommender, MeanScoreRecommender
from evalMethods import evaluate_recall, evaluate_arhr, evaluate_mse


if __name__ == '__main__':
    dataHelper = DataUtils()
    print("loading data...")
    entries = dataHelper.load_data("active1000", num=0) # num is number of files, omit or 0 to load all
    numLoaded = len(entries)
    print(f"loaded {numLoaded} events.")

    # Filter out unhelpful rows
    print("filtering data...")
    filtered_data = dataHelper.filter_data(entries)
    numLost = numLoaded - len(filtered_data)
    print(f"filtered to {len(filtered_data)} events. {numLost} events ({numLost / numLoaded:%}) discarded.")

    # Re-index document and user IDs to start at 0 and be sequential
    print("re-indexing data...")
    indexed_data = dataHelper.index_data(filtered_data)
    print("indexed.")

    # Convert indexed data to a dataframe
    raw_events = dataHelper.get_dataframe(indexed_data)

    # Fix dates, clean up categories, and populate missing data (eventually)
    print("processing data...")
    events, categories = dataHelper.process_data(raw_events)
    print("data processed.")    

    # Generate train-test split
    print("performing train-test split")
    train, test = sklearn.model_selection.train_test_split(events, test_size=0.2, shuffle=False)
    nTest = len(test.index)
    nTrain = len(train.index)

    # Find the train-test split time, and fetch all articles published after that time
    splitTime = train.iloc[-1]["eventTime"] # get the time of the last event in the train set
        # TODO: add a margin, include articles from the same day or week
    splitEventId = train.iloc[-1]["eventId"]
    nTrainArticles = test["documentId"].min()
    nArticles = dataHelper.nextDocumentID - nTrainArticles # number of articles from start of test to end
    #print(newArticles[0], nArticles, newArticles[len(newArticles)-1])
        # the problem here arises because articles are re-indexed by the eventTime of their first interaction, not their publishing time
        # so article 100 may be published after splitTime, while 101 was published before, just because 100 was clicked by someone sooner
    nUsers = dataHelper.nextUserID


    # pass just train set to aggregator
    print(f"Training on {nTrain} events...")
    agg = DataAggregator(categories)
    agg.generateArticleData(train, dataHelper.nextDocumentID)
    agg.generateUserData(train, dataHelper.nextUserID)
    print("Training complete.")

    # Initialize the recommenders on the aggregated train data
    # TODO: Add other recommenders here, perform training here

    meanScore = MeanScoreRecommender(agg.articles, agg.users)
    mostRecent = MostRecentRecommender(agg.articles)
    mostPopular = MostPopularRecommender(agg.articles, splitEventId)

    print(f"Testing on {nTest} events, tracking score for {nArticles} articles...")

    # Initialize verification data, 
    actualClicks = []
    actualScores = np.zeros((nUsers, nArticles))
    # activeTimeMask = np.zeros((nUsers, nArticles)) # to only evaluate MSE for actual user-article pairs
        # activeTimeMask is only needed if recommenders fill out more data than needed
        # since we only evaluate per-event, all predictions and actual scores should be 0 here by default

    # Initialize predicted scores per new article for each user, to test MSE, for recommenders that implement [???]
    # TODO: Add other recommenders here
    meanScores = np.zeros((nUsers, nArticles))

    # Initialize recommended articles per event, to test recall and ARHR, for recommenders that implement predictNextClick
    # TODO: Add other recommenders here
    mostRecentClicks = []
    mostPopularClicks = []

    # These are used to loop over evaluations
    # convenient for any recommenders that have the same format of
    # predictScore(user, article, time) for msePairs
    # predictNextClick(user, time) for clickPairs
    msePairs = ((meanScore, meanScores),)
    clickPairs = ((mostRecent, mostRecentClicks), (mostPopular, mostPopularClicks))

    for eventId, event in test.iterrows():
        relativeEventId = eventId - nTrain # index in the recall-predictions/recommendations
        relativeArticleId = event["documentId"] - nTrainArticles # index in the score-predictions

        # Evaluate scores (activetime)
        if not (relativeArticleId < nArticles):
            print(f'article overflow at {relativeArticleId} with base id {event["documentId"]}')
        if not np.isnan(event["activeTime"]) and relativeArticleId > 0 and relativeArticleId < nArticles: # if event has active time, and article is in our test timeframe
            actualScores[event["userId"], relativeArticleId] = event["activeTime"] # set the actual score
            #activeTimeMask[event["userId"], relativeArticleId] = 1 # mask this for MSE evaulation

            # Get predicted score for each recommender
            # TODO: Add other recommenders here

            for recommender, predictions in msePairs: # optional loop for recommenders with the same format
                p = recommender.predictScore(event["userId"], event["documentId"], event["eventTime"])
                predictions[event["userId"], relativeArticleId] = p
                if np.isnan(p) or not np.isfinite(p): 
                    print(p, event["activeTime"])
        
        # Evaluate clicks
        actualClicks.append(event["documentId"]) # add clicked document corresponding to this event
        # Get top k recommendations for each recommender
        # TODO: Add other recommenders here

        for recommender, predictions in clickPairs:
            predictions.append(recommender.predictNextClick(event["userId"], event["eventTime"], k=10))

        agg.add_event(event) # update our trained data with the new event to improve future predictions
        # TODO: Add other recommenders here

        mostPopular.add_event(event)

        if relativeEventId % 2000 == 0: print('.',end='') # progressbar of sorts
    print('.')
    print("Done")
    
    results = (
        ("MostRecent", None, mostRecentClicks),
        ("MostPopular", None, mostPopularClicks),
        ("MeanScore", meanScores, None),
    )
    resultsTable = []
    for name, scores, clicks in results:
        recall = f"{evaluate_recall(clicks, actualClicks):6.4%}" if clicks is not None else "---"
        arhr = f"{evaluate_arhr(clicks, actualClicks):8f}" if clicks is not None else "---"
        mse = f"{evaluate_mse(scores, actualScores):8f}" if scores is not None else "---"
        resultsTable.append([name, recall, arhr, mse])
    
    print("Results:")
    resultsHeader = ["Method", "Recall", "ARHR", "MSE"]
    print(tabulate(resultsTable, headers=resultsHeader, colalign=("left", "right", "right", "right")))

    # MostRecent does badly, maybe because some?? articles lack a publishtime, making them a non-option
    # TODO: add a firstEventTime to dataUtils that substitutes publishtime when missing?
