import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import concurrent.futures

from tabulate import tabulate
from timeit import default_timer as timer

from dataUtils import DataUtils
from dataAggregator import DataAggregator
from baselines import MostPopularRecommender, MostRecentRecommender, MeanScoreRecommender
from evalMethods import evaluate_recall, evaluate_arhr, evaluate_mse
from contentbased import ContentBasedRecommender


if __name__ == '__main__':
    DATANUM = 9 # number of files, 0 to load all
    K = 10

    dataHelper = DataUtils()
    print("loading data...")
    entries = dataHelper.load_data("active1000", num=DATANUM) 
    numLoaded = len(entries)
    print(f"loaded {numLoaded} events.")

    # Filter out unhelpful rows
    print("filtering data...")
    filtered_data = dataHelper.filter_data(entries)
    numLost = numLoaded - len(filtered_data)
    print(f"filtered to {len(filtered_data)} events. {numLost} events ({numLost / numLoaded:%}) discarded.")

    # Re-index document and user IDs to start at 0 and be sequential, manually
    #print("re-indexing data...")
    #startTime = timer()
    indexed_data = dataHelper.index_data(filtered_data)
    #endTime = timer()
    #print(f"indexed in {endTime - startTime} seconds.")

    # Convert indexed data to a dataframe
    raw_events = dataHelper.get_dataframe(indexed_data)

    # Re-index document and user IDs to start at 0 and be sequential, via factorize
    # half the time (2 seconds instead of 4) but doesnt keep a map back to the original documents/users
    # print("re-indexing data...")
    # startTime = timer()
    # raw_events = dataHelper.factorize_data(raw_events)
    # endTime = timer()
    # print(f"indexed in {endTime - startTime} seconds.")

    # Fix dates, clean up categories, and populate missing data (eventually)
    print("processing data...")
    events, categories = dataHelper.process_data(raw_events)
    print("data processed.")    

    # Generate train-test split
    print("performing train-test split")
    train, test = sklearn.model_selection.train_test_split(events, test_size=0.2, shuffle=False)
    nTest = len(test.index)
    nTrain = len(train.index)
    nUsers = dataHelper.nextUserID
    nArticles = dataHelper.nextDocumentID

    # Find the train-test split time, and fetch all articles published after that time
    splitTime = train.iloc[-1]["eventTime"] # get the time of the last event in the train set
        # TODO: add a margin, include articles from the same day or week
    splitEventId = train.iloc[-1]["eventId"]
    nTrainArticles = train["documentId"].max()
    firstRelevantArticleId = max(test["documentId"].min(), int(nTrainArticles * 0.95))
        # ^ the max() here ensures we DO cut off SOME articles, and assumes that
        # no more than at most the 5% most recent articles (up to ~1k for the full dataset)
        # from before the split are relevant 
    nTestArticles = nArticles - firstRelevantArticleId
        # number of articles from start of test to end (including some from before the split) 


    # Pass just train set to aggregator
    print(f"Aggregating event data over {nTrain} events...")
    agg = DataAggregator(categories, strict=False)

    # multi-thread aggregation for performance boost
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(agg.generateArticleData, train, nArticles)
        executor.submit(agg.generateUserData, train, nUsers)

    # Fill in missing data (in new columns)
    print(f"Filling in missing data...")
    train = dataHelper.fill_missing(train, agg.articles, agg.users)
    agg.generateRatingMatrix(train, nUsers, nArticles)

    # Initialize the recommenders on the aggregated train data
    print(f"Training on {nTrain} events...")
    # TODO: Add other recommenders here, perform training here

    meanScore = MeanScoreRecommender(agg.articles, agg.users)
    mostRecent = MostRecentRecommender(agg.articles, K)
    mostPopular = MostPopularRecommender(agg.articles, splitEventId)

    contentBased = ContentBasedRecommender(train, events, agg.ratings)

    print(f"Testing on {nTest} events, tracking score for {nTestArticles} articles...")

    # Initialize verification data, 
    actualClicks = []
    actualScores = np.zeros((nUsers, nTestArticles))
    # activeTimeMask = np.zeros((nUsers, nTestArticles)) # to only evaluate MSE for actual user-article pairs
        # activeTimeMask is only needed if recommenders fill out more data than needed
        # since we only evaluate per-event, all predictions and actual scores should be 0 here by default

    # Initialize evaluation shorthands
    msePairs = [] # (recommender, array) adds recommender.predictScore to the array every event
    clickPairs = [] # (recommender, list) adds recommender.predictNextClick to the list every event
    adders = [] # (recommender) rusn recommender.add_event(event) to the list every event
    results = [] # ("name", list, array) evaluates list for recall and arhr, and array for mse, and prints with name

    # Initialize each recommenders result data
    meanScores = np.zeros((nUsers, nTestArticles)) # predicted scores per new article for each user, for testing MSE, for recommenders that implement predictScore
    msePairs.append((meanScore, meanScores)) # add recommender + result array to a shorthand for looping
    #clickPairs.append((meanScore, meanScoreClicks)) # mean score baseline does not predict clicks
    #adders.append(meanScore) # mean score baseline does not need to be updated about new events
    results.append(("MeanScore", meanScores, None)) # add recommender + result array to results printout

    contentBasedScores = np.zeros((nUsers, nTestArticles))
    msePairs.append((contentBased, contentBasedScores))

    """ mostRecentClicks = []
    #msePairs.append() # most recent recommender does not predict scores
    clickPairs.append((mostRecent, mostRecentClicks))
    adders.append(mostRecent)
    results.append(("MostRecent", None, mostRecentClicks))

    mostPopularClicks = []
    clickPairs.append((mostPopular, mostPopularClicks))
    adders.append(mostPopular)
    results.append(("MostPopular", None, mostPopularClicks)) """

    contentBasedClicks = []
    #clickPairs.append((contentBased, contentBasedClicks))
    adders.append(contentBased)
    results.append(("ContentBased", contentBasedScores, None))

    for eventId, event in test.iterrows():
        event = dataHelper.fill_single_missing(event, agg.articles, agg.users)

        relativeEventId = eventId - nTrain # index in the recall-predictions/recommendations
        relativeArticleId = event["documentId"] - firstRelevantArticleId # index in the score-predictions

        # Evaluate scores (activetime)
        if not (relativeArticleId < nTestArticles):
            print(f'article overflow at {relativeArticleId} with base id {event["documentId"]}')
        if not np.isnan(event["activeTime"]) and relativeArticleId > 0 and relativeArticleId < nTestArticles: # if event has active time, and article is in our test timeframe
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
            predictions.append(recommender.predictNextClick(event["userId"], event["eventTime"], k=K))

        agg.add_event(event) # update our trained data with the new event to improve future predictions
        #agg.update_rating(event) # only for live ratings
        # TODO: Add other recommenders here
        for recommender in adders:
            recommender.add_event(event)

        if relativeEventId % 2000 == 0 and not relativeEventId == 0:
            print('.',end='',flush=True) # progressbar of sorts
    print('.')
    print("Done")
    
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
