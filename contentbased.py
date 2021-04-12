import json
import os
import pandas as pd
import numpy as np
import gensim # specifically, version 3.8.3
import nltk
from nltk.corpus import stopwords
import re
import time
from heapq import heappush, nlargest

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from rake_nltk import Rake
from gensim.models import Word2Vec

from evalMethods import evaluate_mse, evaluate_arhr, evaluate_recall 

class ContentBasedRecommender:
    def __init__(self, trainEvents, events, rating_matrix, method=1):
        meth = [self.predictNextClickAllTimeHighestExpectedUnread,
                self.predictNextClickMostSimilarToRecent,
                self.predictNextClickMostSimilarToBest,
                self.predictNextClickBestOfBoth]

        nltk.download('stopwords')
        self.similarity_matrix = build_content_based_word2vec_similarity_matrix(events)
        self.rating_matrix = rating_matrix 
        self.user_rated_documents = get_initial_user_rated_documents(trainEvents, rating_matrix.shape[0])
        self.predictNextClick = meth[method] # select method here
        self.known_document_ids = set(trainEvents.get("documentId"))
        self.recencyThreshold = 100


    def predictScore(self, userId, articleId, time):
        #return self.predictions[userId, articleId] #self.actual[userId]
        # get all documentIds that have a proper rating
        rated_documents = list(self.user_rated_documents[userId])
        # get the similarity between the article to predict the score for, and all previous ratings
        sim_array = self.similarity_matrix[articleId, rated_documents] # similarity can be negative sometimes??
        # get the scores for all previously rated articles
        rated_scores = self.rating_matrix[userId, rated_documents]
        # pairwise multiply the above two arrays, and add those scores together (weighted scores)
        summed_rating = np.dot(sim_array, rated_scores)
        # get the total weights used on the scores (the total amount of similarity)
        sim_sum = np.sum(np.absolute(sim_array))
        # return the weighted average
        if (np.isnan(summed_rating) or sim_sum == 0.0 or np.isnan(sim_sum) or np.isnan(summed_rating/sim_sum)):
            return 0.0
        return summed_rating/sim_sum

    def predictNextClickAllTimeHighestExpectedUnread(self, userId, time, k=10): # immense processing costs. unusable
        rated_documents = list(self.user_rated_documents[userId])
        rated_scores = self.rating_matrix[userId, rated_documents]
        viable_documents = {id for id in self.known_document_ids if id > max(self.known_document_ids) - self.recencyThreshold and id not in rated_documents} #self.known_document_ids - self.user_rated_documents[userId] #[doc for doc in self.known_document_ids if doc not in rated_documents]
        heap = []
        for document in viable_documents:
            sim_array = self.similarity_matrix[document, rated_documents]
            summed_rating = np.dot(sim_array, rated_scores)
            sim_sum = np.sum(np.absolute(sim_array))
            if (np.isnan(summed_rating) or sim_sum == 0.0 or np.isnan(sim_sum) or np.isnan(summed_rating/sim_sum)):
                continue
            score = summed_rating/sim_sum
            heappush(heap, (score, document))
        
        return [tup[1] for tup in nlargest(k, heap)]
    
    def predictNextClickMostSimilarToRecent(self, userId, time, k=10): # method 1
        rated_documents = self.user_rated_documents[userId]
        most_recent = max(rated_documents) if rated_documents else max(self.known_document_ids) # hack for efficiency, not truly most recent
        similarity_scores = self.similarity_matrix[most_recent].copy()
        similarity_scores[list(rated_documents)] = 0.0 # make sure previously clicked documents are ignored
        return list(np.argpartition(similarity_scores, -k)[-k:]) # funky way to get document Ids (indices) of k highest scores

    def predictNextClickMostSimilarToBest(self, userId, time, k=10): # method 2
        rated_documents = list(self.user_rated_documents[userId])
        rated_scores = self.rating_matrix[userId, rated_documents]
        top_rated = rated_documents[list(rated_scores).index(max(rated_scores))] if rated_scores.size else max(self.known_document_ids)
        similarity_scores = self.similarity_matrix[top_rated].copy()
        similarity_scores[rated_documents] = 0.0
        return list(np.argpartition(similarity_scores, -k)[-k:])

    def predictNextClickBestOfBoth(self, userId, time, k=10): # method 3: take options from 1 and 2, then sort by predicted scores
        rated_documents = list(self.user_rated_documents[userId])
        rated_scores = self.rating_matrix[userId, rated_documents]

        if not (rated_documents and rated_scores.size):
            return self.predictNextClickMostSimilarToBest(userId, time, k)
        
        most_recent = max(rated_documents) 
        top_rated = rated_documents[list(rated_scores).index(max(rated_scores))] 
        document_pair = [most_recent, top_rated]
        score_pair = [rated_scores[rated_documents.index(most_recent)], max(rated_scores)] #rated_scores[document_pair]

        closest_to_recent = self.predictNextClickMostSimilarToRecent(userId, time, k)
        closest_to_best = self.predictNextClickMostSimilarToBest(userId, time, k)
        viable_documents = set(closest_to_recent).union(set(closest_to_best))

        heap = []
        for document in viable_documents:
            sim_array = self.similarity_matrix[document, document_pair]
            summed_rating = np.dot(sim_array, score_pair)
            sim_sum = np.sum(np.absolute(sim_array))
            if (np.isnan(summed_rating) or sim_sum == 0.0 or np.isnan(sim_sum) or np.isnan(summed_rating/sim_sum)):
                continue
            score = summed_rating/sim_sum
            heappush(heap, (score, document))
        return [tup[1] for tup in nlargest(k, heap)]

    def add_event(self, event):
        # if event contains an activetime, add the documentId to the users confirmed rated documents
        if isRating(event["activeTime"]):
            self.user_rated_documents[event["userId"]].add(event["documentId"])
        self.known_document_ids.add(event["documentId"])
        
def isRating(activeTime):
    return not (np.isnan(activeTime) or activeTime == 0.0)

def get_initial_user_rated_documents(trainEvents, nUsers):
    # initialize a set (quick lookup, no duplicates) for each userId (you can be certain that every userId exists)
    user_rated_documents = tuple([set() for i in range(nUsers)])
    # get all documents each user have clicked on, turn them into a set.
    df = trainEvents[["userId", "documentId"]].groupby("userId").agg(lambda id: set(id)).reset_index()
    for _, row in df.iterrows():
        user_rated_documents[row["userId"]].update(row["documentId"])
    return user_rated_documents

def remove_stopwords(words):
    set_stopwords = set(stopwords.words('norwegian'))
    set_stopwords.add('')
    words = [x for x in words if not x in set_stopwords]
    return words

def build_content_based_word2vec_similarity_matrix(events):
    def modeAgg(x): 
        m = pd.Series.mode(x)
        return m.values[0] if not m.empty else None
        
    df = events[['categories', 'documentId', 'title']]
    # take the most common version of categories or title for that document:
    df = df.groupby('documentId').agg(modeAgg).reset_index()

    # combine category and title, doing some preprocessing
    df['title'] = df['title'].fillna("").astype('str')
    titlecat = df['categories'].map(lambda categoryArr: ' '.join(categoryArr if categoryArr is not None else [])) + ' ' + df['title']
    titlecat = (titlecat.map(lambda string: re.sub('[^a-zøæåA-ZØÆÅ]+', ' ', string).lower())
                        .str.split(' ')
                        .map(lambda words: remove_stopwords(words)))

    # build word embeddings model
    word_vec_model = gensim.models.Word2Vec(titlecat, min_count = 1,  size = 100, window = 5) 

    # print('number of unique words in our word vocabulary : ' + str(len(list(word_vec_model.wv.vocab))))
    word_vec_model.init_sims(replace=True)
    
    # dictionary with vector values for each word
    vectors_dict = word_vec_model.wv
    
    # convert each word in titlecat into their vector representation
    vectors_array = titlecat.apply(lambda words: [vectors_dict[x] for x in words]).to_numpy()

    # max_len = max(len(row) for row in vectors_array) #max number of words in titlecats is 31
    # print(max_len)

    # insert vectors into padded matrix
    padded_matrix = np.zeros((vectors_array.shape[0], 31, 100))
    # A possible reason why all rows are to some degree similar are the empty remainder of the 31 max len words for each row
    # eg: 0 is similar to 0 
    for x in range(vectors_array.shape[0]):
        for y in range(len(vectors_array[x])):
            padded_matrix[x,y,:100] = vectors_array[x][y]


    #concatenate the padded matrix so that we only get a single vector for each row eg titlecat row
    concatinated_matrix = np.asmatrix([np.concatenate(arrays) for arrays in padded_matrix])

    #build similarity matrix for each titlecat row where the index [0][0] is the similarity of documentId with itself
    similarity_matrix = cosine_similarity(concatinated_matrix)
    return similarity_matrix
 