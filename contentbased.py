import json
import os
import pandas as pd
import numpy as np
import gensim 
import nltk
from nltk.corpus import stopwords
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from rake_nltk import Rake
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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


def build_content_based_word2vec_similarity_matrix(df):

    # Remove front page events adressa.no
    # also consist of all titles that have null value
    df = df.dropna(subset=['documentId'])

    # Drop duplicate 
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df.drop_duplicates(subset=['documentId'], inplace=True)

    # remap document and userids to integers starting from 0
    df['userId'] = pd.factorize(df['userId'])[0]
    df['documentId'] = pd.factorize(df['documentId'])[0]
    
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
    
    # array of phrases from each titlecat row
    phrase_array = df['titlecat'].to_numpy()
    
    # vector array for each phrase in each titlecat row
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
    print(similarity_matrix[0][0])
    print('similarity matrix shape: '+ str(similarity_matrix.shape))
    return df, similarity_matrix



    

if __name__ == '__main__':
    # df=load_data('active1000')
    df=load_data_test()
    
    build_content_based_word2vec_similarity_matrix(df)

