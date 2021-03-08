import json
import os
import pandas as pd
import numpy as np
import gensim 
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
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


def statistics(df):
    """
        Basic statistics based on loaded dataframe
    """
    total_num = df.shape[0]
    
    print("Total number of events(front page incl.): {}".format(total_num))
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    df_ref = df[df['documentId'].notnull()]
    num_act = df_ref.shape[0]
    
    print("Total number of events(without front page): {}".format(num_act))
    num_docs = df_ref['documentId'].nunique()
    
    print("Total number of documents: {}".format(num_docs))
    print('Sparsity: {:4.3f}%'.format(float(num_act) / float(1000*num_docs) * 100))
    df_ref.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    print("Total number of events(drop duplicates): {}".format(df_ref.shape[0]))
    print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df_ref.shape[0]) / float(1000*num_docs) * 100))
    
    user_df = df_ref.groupby(['userId']).size().reset_index(name='counts')
    print("Describe by user:")
    print(user_df.describe())

def evaluate(pred, actual, k):
    """
    Evaluate recommendations according to recall@k and ARHR@k
    """
    total_num = len(actual)
    tp = 0.
    arhr = 0.
    for p, t in zip(pred, actual):
        if t in p:
            tp += 1.
            arhr += 1./float(p.index(t) + 1.)
    recall = tp / float(total_num)
    arhr = arhr / len(actual)
    print("Recall@{} is {:.4f}".format(k, recall))
    print("ARHR@{} is {:.4f}".format(k, arhr))



def content_processing(df):
    """
        Remove events which are front page events, and calculate cosine similarities between
        items. Here cosine similarity are only based on item category information, others such
        as title and text can also be used.
        Feature selection part is based on TF-IDF process.
    """
    df = df[df['documentId'].notnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df['category'] = df['category'].str.split('|')
    df['category'] = df['category'].fillna("").astype('str')

    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True) 
    	

    #select features/words using TF-IDF 
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    print('Dimension of feature vector: {}'.format(tfidf_matrix.shape))

    #measure similarity of two articles with cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    print("Similarity Matrix:")
    print(cosine_sim[:4, :4])
    return cosine_sim, df

def content_recommendation(df, k=20):
    """
        Generate top-k list according to cosine similarity
    """
    cosine_sim, df = content_processing_word2vec(df)
    #cosine_sim, df = content_processing(df)
    df = df[['userId','time', 'tid', 'title', 'category']]
    df.sort_values(by=['userId', 'time'], ascending=True, inplace=True)
    print(df[:20]) # see how the dataset looks like
    pred, actual = [], []
    puid, ptid1, ptid2 = None, None, None
    for row in df.itertuples():
        uid, tid = row[1], row[3]
        if uid != puid and puid != None:
            idx = ptid1
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            sim_scores = [i for i,j in sim_scores]
            pred.append(sim_scores)
            actual.append(ptid2)
            puid, ptid1, ptid2 = uid, tid, tid
        else:
            ptid1 = ptid2
            ptid2 = tid
            puid = uid
    
    evaluate(pred, actual, k)


def nlp(x,r):
    r.extract_keywords_from_text(x)
    return r.get_ranked_phrases()


def content_based_word2vec(df):

    #Remove front page events adressa.no
    #also consist of all titles that have null value
    df = df.dropna(subset=['documentId'])

    #Drop duplicate 
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df.drop_duplicates(subset=['documentId'], inplace=True)

    #remap document and userids to integers starting from 0
    df['userId'] = pd.factorize(df['userId'])[0]
    df['documentId'] = pd.factorize(df['documentId'])[0]
    
    df['category'] = df['category'].fillna("").astype('str')
    df['title'] = df['title'].fillna("").astype('str')
    df['category'] = df['category'].str.replace('|',' ')

    df['titlecat'] = df['category'] + ' ' + df['title']
    # df['titlecat'] = df['titlecat'].apply(lambda string: re.sub('[^a-zA-Z0-9 øæå]+', ' ', string).lower())
    # df['titlecat'] = df['titlecat'].str.split(' ')

    #tokenizeation, removing stop words, extracting key phrases
    r = Rake(language='norwegian')
    df['titlecat'] = df['titlecat'].apply(lambda x: nlp(x,r))
    print(df[:20])

    word_vec_model = gensim.models.Word2Vec(df['titlecat'], min_count = 1,  size = 100, window = 5) 

    
    # print(list(word_vec_model.wv.vocab))
    print(len(list(word_vec_model.wv.vocab)))
    word_vec_model.init_sims(replace=True)
    
    vectors = word_vec_model.wv
    phrase_matrix = df['titlecat'].to_numpy()
    print(phrase_matrix[0])
    # for row in phrase_matrix:
        

    sim_matrix = np.dot(vectors.syn0norm, vectors.syn0norm.T)

    print(sim_matrix.shape)
    print(sim_matrix[:5])
    print(len(sim_matrix[0]))

    

if __name__ == '__main__':
    #df=load_data('active1000')
    df=load_data_test()
    
    ###### Get Statistics from dataset ############
    #print("Basic statistics of the dataset...")
    #statistics(df)

    ###### Recommendations based on Content-based Method (Cosine Similarity) ############
    #print("Recommendation based on content-based method...")
    #content_recommendation(df, k=20)
    content_based_word2vec(df)

