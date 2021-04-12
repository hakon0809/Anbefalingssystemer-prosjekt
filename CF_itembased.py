import sys
import pandas as pd
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import implicit
import sklearn.model_selection


from project_example import load_data, load_dataset
from evalMethods import evaluate_recall, evaluate_arhr, evaluate_mse

# This file is incomplete

def shape_data(df):
    """
        Shape data to [activetime, user, article]
    """
    df = df[~df['documentId'].isnull()]
    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
    df = df.sort_values(by=['userId', 'time'])
    df = df.drop(['eventId', 'category', 'title', 'url', 'publishtime', 'time'], axis=1)
    df.columns = ['activetime', 'user', 'article']
    df['activetime'].fillna(value=df['activetime'].mean()/3, inplace=True)
    # df['activetime'].fillna(value=0, inplace=True)   
    # df['activetime'].fillna(value= df.groupby(['user']).activetime.mean(), inplace=True)
    
    return df

def implicit_als(df):
    
    df['user'] = df['user'].astype("category")
    df['article'] = df['article'].astype("category")
    
    new_user = df['user'].values[1:] != df['user'].values[:-1]
    new_user = np.r_[True, new_user]
    df['user_id'] = np.cumsum(new_user)

    df['article_id'] = df['article'].cat.codes

    sparse_item_user = sparse.csr_matrix((df['activetime'].astype(float), (df['article_id'], df['user_id'])))
    sparse_user_item = sparse.csr_matrix((df['activetime'].astype(float), (df['user_id'], df['article_id'])))

    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

    # Calculate confidence with given alpha value.
    alpha_val = 40
    data_conf = (sparse_item_user * alpha_val).astype('double')

    #Fit the model
    model.fit(data_conf)


    #------------------
    # ITEM SIMILARITY
    #------------------

    # item_id = 9983      #959778
    item_id = 14515   #962662
    # item_id = 3336    #962665
    # item_id = 771     #867734
    n_similar = 10

    print(df.article.loc[df.article_id == item_id].iloc[0])

    # Use implicit to get similar items.
    similar = model.similar_items(item_id, n_similar)


    pred = []
    # Print the names of our most similar articles
    for item in similar:
        idx, score = item
        # print(article_names.loc[article_names.documentId == df.article.loc[df.article_id == idx].iloc[0]].iloc[0].title)
        # pred.append(article_names.loc[article_names.documentId == df.article.loc[df.article_id == idx].iloc[0]].iloc[0].title)
        pred.append(df.article.loc[df.article_id == idx].iloc[0])

    return pred

    #------------------------------
    # CREATE USER RECOMMENDATIONS
    #------------------------------

    # # Create recommendations for a given user_id
    # user_id = 1

    # # Use the implicit recommender.
    # recommended = model.recommend(user_id, sparse_user_item)

    # articles = []
    # scores = []

    # # Get article names from ids
    # for item in recommended:
    #     idx, score = item
    #     articles.append(df.article.loc[df.article_id == idx].iloc[0])
    #     scores.append(score)

    # # Create a dataframe of articles and scores
    # recommendations = pd.DataFrame({'article': articles, 'score': scores})

    # # Get article name from article_id
    # article_titles = []
    # for article in recommendations['article']:
    #     article_titles.append(article_names.loc[article_names.documentId == article].iloc[0].title)

    # print(recommendations)
    # print(article_titles)
    


df=load_data("active1000")

# df = pd.read_csv("df_with_average_activetime.csv")

data = shape_data(df)

train, test = sklearn.model_selection.train_test_split(data, test_size=0.2, shuffle=False)


pred = implicit_als(train)  #returns list of similar articles to 'c2c945f8abcb9c7e31886d068c8c41ce8d37a7e4'
print(pred)

# actual = test['article'].tolist()

# print(evaluate_arhr(pred,actual))


