import json
import os
import pandas as pd
import numpy as np
import gensim 
import nltk
from nltk.corpus import stopwords
import re
import time

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from rake_nltk import Rake
from gensim.models import Word2Vec

from evalMethods import evaluate_mse, evaluate_arhr, evaluate_recall 

nltk.download('stopwords')

pd.options.display.max_columns = 10
pd.options.display.max_rows = 50

def load_data(path):
    """
        Load events from files and convert to dataframe.
    """
    map_lst=[]
    for f in os.listdir(path):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if not obj is None:
                    map_lst.append(obj)
    df = pd.DataFrame(map_lst)

    df_test = df[df.index % 10 == 0]
    df_test.to_csv('testfile.csv')
    
    return df

def load_data_test():
    """
        Load events from csv and convert to dataframe.
    """
    return pd.read_csv('testfile.csv')

def load_data_average():
    df = pd.read_csv('df_with_average_activetime.csv')
    df_test = df[df.index % 10 == 0]
    df_test.to_csv('testfileavg.csv')
    return df

def load_data_avg_test():
    """
        Load events from csv and convert to dataframe.
    """
    return pd.read_csv('testfileavg.csv')


def nlp(x,r):
    r.extract_keywords_from_text(x)
    return r.get_ranked_phrases()

#tokenizeation, removing stop words, extracting key phrases
# r = Rake(language='norwegian')
# df['titlecat'] = df['titlecat'].apply(lambda x: nlp(x,r))

def remove_stopwords(words):
    set_stopwords = set(stopwords.words('norwegian'))
    set_stopwords.add('')
    words = [x for x in words if not x in set_stopwords]
    return words


def remap_df(df):
    # Remove front page events adressa.no
    # which also consist of all titles that have null value
    # print(len(df.index))
    # df = df.dropna(subset=['documentId'])
    # print(len(df.index))

    # remap document and userIds to integers starting from 0
    df['documentId'] = pd.factorize(df['documentId'])[0]
    df['userId'] = pd.factorize(df['userId'])[0]

    #fill null active time with 3.6
    df['activeTime'] = df['activeTime'].fillna(0).astype('float')
    return df

def build_content_based_word2vec_similarity_matrix(df):

    df = df[['category', 'documentId', 'title']]

    # Drop duplicate 
    # df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df.drop_duplicates(subset=['documentId'], inplace=True)

    
    # combine category and title, doing some preprocessing
    df['category'] = df['category'].fillna("").astype('str')
    df['title'] = df['title'].fillna("").astype('str')
    df['category'] = df['category'].str.replace('|',' ')

    df['titlecat'] = df['category'] + ' ' + df['title']
    df['titlecat'] = df['titlecat'].apply(lambda string: re.sub('[^a-zøæåA-ZØÆÅ]+', ' ', string).lower())
    df['titlecat'] = df['titlecat'].str.split(' ')
    df['titlecat'] = df['titlecat'].apply(lambda words: remove_stopwords(words))

    # build word embeddings model
    word_vec_model = gensim.models.Word2Vec(df['titlecat'], min_count = 1,  size = 100, window = 5) 

    
    print('number of unique words in our word vocabulary : ' + str(len(list(word_vec_model.wv.vocab))))
    word_vec_model.init_sims(replace=True)
    
    # dictionary with vector values for each word
    vectors_dict = word_vec_model.wv
    print(len(df['titlecat'].index))
    # array of phrases from each titlecat row
    phrase_array = df['titlecat'].to_numpy()
    
    
    # convert each word in titlecat into their vector representation
    vectors_array = df['titlecat'].apply(lambda words: [vectors_dict[x] for x in words]).to_numpy()

    # max_len = max(len(row) for row in vectors_array) #max number of words in titlecats is 31
    # print(max_len)

    # insert vectors into padded matrix
    padded_matrix = np.zeros((vectors_array.shape[0], 31, 100))
    for enu, row in enumerate(vectors_array):
        padded_matrix[enu, :len(row)] += row

    print('padded matrix shape: ' + str(padded_matrix.shape))

    #concatenate the padded matrix so that we only get a single vector for each row eg titlecat row
    concatinated_matrix = np.asmatrix([np.concatenate(arrays) for arrays in padded_matrix])

    print('concatinated matrix shape' + str(concatinated_matrix.shape))

    #build similarity matrix for each titlecat row where the index [0][0] is the similarity of documentId with itself
    similarity_matrix = cosine_similarity(concatinated_matrix)
    # print(similarity_matrix[:5,:5])
    print('similarity matrix shape: '+ str(similarity_matrix.shape))
    return similarity_matrix

def build_rating_matrix(df):
    #ratingmatrix(users,items)
    df = df[['userId','documentId', 'activeTime']]
    nr_users = len(df['userId'].drop_duplicates())
    nr_documents = len(df['documentId'].drop_duplicates())
    rating_matrix = np.zeros((nr_users,nr_documents))
    #TODO NEEDS BETTER FIX
    df['activeTime'] = df['activeTime'].fillna(65.0)
    #can be done faster using vectorization maybe
    for index, row in df.iterrows():
            rating_matrix[int(row['userId']), int(row['documentId'])] += row['activeTime']


    return rating_matrix

def train_test_split(ratings, fraction=0.2):
    """Leave out a fraction of dataset for test use"""
    test = np.zeros(ratings.shape) # makes a empty matching list to ratings_table
    train = ratings.copy() # makes a copy of the rating_table
    for user in range(ratings.shape[0]): # Loop on the users in the ratings_table
        # size(Int) = is the amount of ratings * fraction for that specific user
        size = int(len(ratings[user, :].nonzero()[0]) * fraction) 
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=size, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    return train, test

def predict_active_time(df,similarity_matrix, rating_matrix):
    #sum(sigma (similarity * activetime))/sigma(similarity)
    # similar to item-based neighborhood models - Adjusted cosine similarity
    train, test = train_test_split(rating_matrix)
    pred = []
    actual = []
    
    #predict scores for all users based on the 20% test split for each user
    for user in range(rating_matrix.shape[0]):
        #get training indexes and actual active time values
        train_index_array = train[user, :].nonzero()[0]
        train_value_array = rating_matrix[user,train_index_array]

        #get test indexes and actual active time values for test
        test_index_array = test[user, :].nonzero()[0]
        test_value_array = rating_matrix[user,test_index_array]
        actual.append(test_value_array)

        #loop through test index array to predict values for those articles
        for index in test_index_array:
            # 1d cosine similarity array between all train active_time scores and target test index
            sim_array = similarity_matrix[train_index_array,index]
            #summed product between cosine similarities and train active_time score
            summed_rating = np.dot(sim_array, train_value_array)
            #sum of train cosine similarity to target test index
            sim_sum = np.sum(np.absolute(sim_array))
            #predicted score
            pred.append(summed_rating/sim_sum)

    actual = [x for row in actual for x in row] 
    print(pred[:10])
    print(actual[:10])


    return pred, actual

#recommend most similar to item read with highest active time
def content_recommendation_m1(rating_matrix, similarity_matrix, train, test):
    pred = []
    actual = []

    for user in range(rating_matrix.shape[0]):
        train_index_array = train[user, :].nonzero()[0]
        train_value_array = rating_matrix[user,train_index_array]

        test_index_array = test[user, :].nonzero()[0]
        test_value_array = rating_matrix[user,test_index_array]
        actual.append(test_index_array)

        highest_train_active_time_doc = train_index_array[np.argmax(train_value_array)]

        nr_predictions = len(test_index_array) + 1
        sim_array = similarity_matrix[highest_train_active_time_doc,:]
        user_pred = []
        while nr_predictions > 0:
            prediction = np.argmax(sim_array)
            sim_array[prediction] = 0

            user_pred.append(prediction)
            nr_predictions -= 1

        # slice first element to remove similarity between the document itself 
        pred.append(user_pred[1:])
        # pred += [user_pred[1:] for i in range(len(test_index_array))]


    actual = [x for row in actual for x in row] 
    return pred, actual
    # print(result)

#recommend most similar to item with highest active time, not previously read
def content_recommendation_m2(rating_matrix, similarity_matrix, train, test):
    pred = []
    actual = []
    
    for user in range(rating_matrix.shape[0]):
        train_index_array = train[user, :].nonzero()[0]
        train_value_array = rating_matrix[user,train_index_array]

        test_index_array = test[user, :].nonzero()[0]
        test_value_array = rating_matrix[user,test_index_array]
        actual.append(test_index_array)

        highest_train_active_time_doc = train_index_array[np.argmax(train_value_array)]

        nr_predictions = len(test_index_array) + 1
        sim_array = similarity_matrix[highest_train_active_time_doc,:]
        user_pred = []
        while nr_predictions > 0:
            prediction = np.argmax(sim_array)
            sim_array[prediction] = 0
            if not prediction in train_index_array:
                user_pred.append(prediction)
                nr_predictions -= 1
            else:
                pass
        # slice first element to remove similarity between the document itself 
        pred.append(user_pred[1:])
        # pred += [user_pred[1:] for i in range(len(test_index_array))]

    actual = [x for row in actual for x in row] 
    return pred, actual

#recommend k most similar to item with highest active time, not previously read
def content_recommendation_m3(k, rating_matrix, similarity_matrix, train, test):
    pred = []
    actual = []
    
    for user in range(rating_matrix.shape[0]):
        train_index_array = train[user, :].nonzero()[0]
        train_value_array = rating_matrix[user,train_index_array]

        test_index_array = test[user, :].nonzero()[0]
        test_value_array = rating_matrix[user,test_index_array]
        actual.append(test_index_array)

        highest_train_active_time_doc = train_index_array[np.argmax(train_value_array)]

        nr_predictions = len(test_index_array) * k + 1
        sim_array = similarity_matrix[highest_train_active_time_doc,:]
        user_pred = []
        while nr_predictions > 0:
            prediction = np.argmax(sim_array)
            sim_array[prediction] = 0
            if not prediction in train_index_array:
                user_pred.append(prediction)
                nr_predictions -= 1
            else:
                pass
        # slice first element to remove similarity between the document itself 
        pred.append(user_pred[1:])
        # pred += [user_pred[1:] for i in range(len(test_index_array))]

    actual = [x for row in actual for x in row] 
    return pred, actual

def content_recommendation(rating_matrix, similarity_matrix):
    #TODOES
    #recommend most similar to last item read
    #recommend most similar to k last items read
    #bayes classifier maybe

    train, test = train_test_split(rating_matrix)
    pred, actual = content_recommendation_m1(rating_matrix, similarity_matrix, train, test)
    print('Metode 1')
    print(evaluate_recall(pred, actual))
    pred, actual = content_recommendation_m2(rating_matrix, similarity_matrix, train, test)
    print('Metode 2')
    print(evaluate_recall(pred, actual))
    pred, actual = content_recommendation_m3(10,rating_matrix, similarity_matrix, train, test)
    # print('Metode 3')
    # print(evaluate_recall(pred, actual))
    return

    

if __name__ == '__main__':
    # df = load_data('active1000')
    # df = load_data_test()
    df = load_data_average()
    # df = load_data_avg_test()

    # df = remap_df(df)
    similarity_matrix = build_content_based_word2vec_similarity_matrix(df)
    rating_matrix = build_rating_matrix(df)
    pred, actual = predict_active_time(df, similarity_matrix, rating_matrix)
    print(evaluate_mse(pred, actual))


    content_recommendation(rating_matrix,similarity_matrix)

 