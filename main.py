import pandas as pd
import numpy as np
import sklearn

from dataUtils import DataUtils
from dataAggregator import DataAggregator
from baselines import MostPopularRecommender, MostRecentRecommender
from evalMethods import evaluate_recall, evaluate_arhr, evaluate_mse

# def statistics(df):
#     """
#         Basic statistics based on loaded dataframe
#     """
#     total_num = df.shape[0]
    
#     print(f"Total number of events(front page incl.): {total_num}")
#     df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
#     df_ref = df[df['documentId'].notnull()]
#     num_act = df_ref.shape[0]
    
#     print(f"Total number of events(without front page): {num_act}")
#     num_docs = df_ref['documentId'].nunique()
    
#     print(f"Total number of documents: {num_docs}")
#     print('Sparsity: {:4.3f}%'.format(float(num_act) / float(1000*num_docs) * 100))
#     df_ref.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
#     print(f"Total number of events(drop duplicates): {df_ref.shape[0]}")
#     print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df_ref.shape[0]) / float(1000*num_docs) * 100))
    
#     user_df = df_ref.groupby(['userId']).size().reset_index(name='counts')
#     print("Describe by user:")
#     print(user_df.describe())

if __name__ == '__main__':
    dataHelper = DataUtils()
    print("loading data...")
    entries = dataHelper.load_data("active1000")
    print("data loaded.")

    # Filter out unhelpful rows
    print("filtering data...")
    filtered_data = dataHelper.filter_data(entries)
    print("data filtered.")

    # Re-index document and user IDs to start at 0 and be sequential
    # acts in-place, kind of confusing, possibly a TODO for later
    indexed_data = dataHelper.index_data(filtered_data)

    # Convert indexed data to a dataframe
    raw_events = dataHelper.get_dataframe(indexed_data)

    # Fix dates, clean up categories, and populate missing data (eventually)
    events = dataHelper.process_data(raw_events)

    # TODO: TRAIN TEST SPLIT HERE
    # pass just train set to aggregator
    agg = DataAggregator()
    agg.generateArticleData(events, dataHelper.nextDocumentID)
    agg.generateUserData(events, dataHelper.nextUserID)

    recommender = MostPopularRecommender(agg.articles)

    # Rough testing of one alg
    test = events.tail(20)
    
    pred = []
    actual = []
    for i, event in events.iterrows():
        actual.append(event["documentId"])
        pred.append(recommender.predictNextClick(event["userId"], event["eventTime"]))
        # TODO: make aggregator step forward event-by-event
    recall = evaluate_recall(pred, actual)
    arhr = evaluate_arhr(pred, actual)
    print(f"recall: {recall}, arhr: {arhr}")

    #train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, shuffle=False)
