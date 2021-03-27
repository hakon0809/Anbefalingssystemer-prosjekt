import sys
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import implicit

from project_example import load_data, load_dataset, train_test_split


def shape_data(df):
    """
        Convert dataframe to user-item-interaction matrix, which is used for 
        Matrix Factorization based recommendation.
        In rating matrix, clicked events are refered as 1 and others are refered as 0.
    """
    df = df[~df['documentId'].isnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df = df.sort_values(by=['userId', 'time'])
    df = df.drop(['eventId', 'category', 'title', 'url', 'publishtime', 'time'], axis=1)
    # print(df[40:48])
    df.columns = ['activetime', 'user', 'article']
    df['activetime'].fillna(value=df['activetime'].mean()/3, inplace=True)
    # df['activetime'].fillna(value=0, inplace=True)
    # df['activetime'].fillna(value= df.groupby(['user']).activetime.mean(), inplace=True)
    # print(df[:5])
    n_users = df['user'].nunique()
    n_items = df['article'].nunique()

    # ratings = np.zeros((n_users, n_items))
    # new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    # new_user = np.r_[True, new_user]
    # df['uid'] = np.cumsum(new_user)
    # item_ids = df['documentId'].unique().tolist()
    # user_ids = df['userId'].unique().tolist()

    # active_times = df['activetime'].unique().tolist()
    # print(active_times[:10])
    
    return df

def implicit_als(df):

    df['user'] = df['user'].astype("category")
    df['article'] = df['article'].astype("category")

    new_user = df['user'].values[1:] != df['user'].values[:-1]
    new_user = np.r_[True, new_user]
    df['user_id'] = np.cumsum(new_user)

    # df['user_id'] = df['user_id'].cat.codes
    df['article_id'] = df['article'].cat.codes

    # print(df['article_id'][40:48])
    # print(df[1800:1840])


    sparse_item_user = sparse.csr_matrix((df['activetime'].astype(float), (df['article_id'], df['user_id'])))
    sparse_user_item = sparse.csr_matrix((df['activetime'].astype(float), (df['user_id'], df['article_id'])))

    # print(sparse_item_user[:10])

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

    # Calculate the confidence by multiplying it by our alpha value.
    alpha_val = 15
    data_conf = (sparse_item_user * alpha_val).astype('double')

    #Fit the model
    model.fit(data_conf)

    item_id = 9983      #959778
    # item_id = 14515   #962662
    # item_id = 3336    #962665
    # item_id = 771     #867734
    n_similar = 10

    # Use implicit to get similar items.
    similar = model.similar_items(item_id, n_similar)

    # Print the names of our most similar articles
    for item in similar:
        idx, score = item
        print(df.article.loc[df.article_id == idx].iloc[0])
    

    #------------------------------
    # CREATE USER RECOMMENDATIONS
    #------------------------------

    # Create recommendations for user with id 4
    user_id = 4

    # Use the implicit recommender.
    recommended = model.recommend(user_id, sparse_user_item)

    articles = []
    scores = []

    # Get article names from ids
    for item in recommended:
        idx, score = item
        articles.append(df.article.loc[df.article_id == idx].iloc[0])
        scores.append(score)

    # Create a dataframe of articles and scores
    recommendations = pd.DataFrame({'article': articles, 'score': scores})

    # article_titles = []
    # for article in recommendations:
    #     art, score = article
    #     article_titles.append(df.article.loc[df.article == art].iloc[0])
    # print(article_titles)

    print(recommendations)




# 7e98f8a1a50a409a25831be225e01e261dfe04fc
# 9e1a8fe1f3370099aff8529a9381fdabd7d5d74a
# b1b9d6843e7e9a5b146979c3cd1303cfacae584d
# b28e7c163c39941aa1cbd0b7b3a821576771f893
    


df=load_data("active1000")


data = shape_data(df)
implicit_als(data)
# print(data[:10])