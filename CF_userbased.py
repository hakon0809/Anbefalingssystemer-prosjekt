from datetime import time
import os
import json
import numpy as np
import pandas as pd
import matplotlib as plt

from timeit import default_timer as timer
from pandas._libs.tslibs import Timestamp

from pandas.tseries.offsets import Second
from surprise.model_selection import validation

import dataUtils
import project_example as pe
import ExplicitMF as mf

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
    return pd.DataFrame(map_lst)

def load_one_file(path):
    """
        Less loading time with working on smaller dataset
        Will always load file 20170101 or a specific file given by
        path. Load events from files and convert to dataframe.
    """
    map_lst=[]
    file_name=os.path.join(path)
    if os.path.isfile(file_name):
        for line in open(file_name):
            obj = json.loads(line.strip())
            if not obj is None:
                map_lst.append(obj)
    return pd.DataFrame(map_lst) 

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

def remove_all_none_values(df):
    '''
        Basically removing 2/3 of the dataset....
    '''

    amount_before = df.shape[0]
    df = df[df['eventId'].notnull()]
    df = df[df['documentId'].notnull()]
    df = df[df['userId'].notnull()]
    amount_after = df.shape[0] 
    print("The amount of the dataset that was discarded: ", str(100 - int((amount_after/amount_before)*100)), "%")

    return df

def replace_NaN_w_0(df, column=False):
    if column:
        df[str(column)] = df[str(column)].replace(np.nan, 0)
        return df[str(column)]
    else:
        df = df.replace(np.nan, 0)
        return df

def centered_cosine(df):
    '''
        Centered Cosine
        1. We take the mean of user's activeTime.
        2. Subtract that mean from all individual activeTime divided by the total number of activeTime by the user.
        3. For all the activeTimes where there is no time(NaN), we replace it with 0.
    '''
    # We create view a list of all mean of activetimes that a user had. We then loop on that list by userId.
    #   userId:                                            mean_of_that_user:
    #  'cx:10k2wzm1b3jsk2y5ym1utnjo97:2kefzqaxe9jx7'        13.75
    #  'cx:11asuyo07mbi13b849lp87hlje:1i9z1hu1xx694'        75.85714285714286
    #   ...                                                 ...
    for userId, mean_of_that_user in df.groupby(['userId']).activeTime.mean().iteritems():
        mean_of_that_user = round(mean_of_that_user, 2)
        # Access each individual activeTime of a specific userId
        activeTimeS = df.loc[df.userId == userId, ['activeTime']]

        # Loop on all activitimes of the specific userId, updates its activeTime
        for i in range(len(activeTimeS)):
            if activeTimeS.iloc[i].notna().bool() == True: # NaN check
               activeTimeS.iloc[i] = round(activeTimeS.iloc[i] - mean_of_that_user, 2)
            else: 
                activeTimeS.iloc[i] = 0.0         
        df.loc[df.userId == userId, ['activeTime']] = activeTimeS

    return df

def replace_none_activetime_with_average(df):
    ''' 
        Finding the documentIds with None activeTime. Then find if there exist others in the system
        that have looked at those specific documentIds with a valid activeTime. Replace the none activeTime
        with average time of activeTime found.

    '''
    # A DF of values
    documentIds_value = df.loc[df.activeTime.notnull(), ['documentId','activeTime']]
    # A DF of none values
    documentIds_null = df.loc[df.activeTime.isnull(), ['documentId','activeTime']]

    # Find all documentIds with activeTime == None/NaN/null/whatever
    for docuId in documentIds_null['documentId']:
        # Find all activeTimes of the first documentId==None. NB Here it is important that we find only real values
        new_documentIds_value = documentIds_value[documentIds_value.documentId == docuId].activeTime
        
        document_list_sum = int(new_documentIds_value.sum())
        document_list_len = int(len(new_documentIds_value))

        if document_list_len <= 0 | document_list_sum <= 0:
            pass
        else:
            documentIds_null.loc[documentIds_null.documentId == docuId, 'activeTime'] = int(document_list_sum / document_list_len)
        
    df.loc[df.activeTime.isnull(), ['documentId','activeTime']] = documentIds_null
    
    df.to_csv('df_with_average_activetime_kkkk.csv', index=False)

    #new_df = pd.DataFrame(documentIds_null) 
    #for id, chunk in enumerate(np.array_split(new_df, 4)): 
    #    chunk.to_csv(f'new_df_avg_activetime_{id}.csv', index=False) 

    return df

def preprocessing_data(df):

    df = df[df['eventId'].notnull()]
    df = df[df['documentId'].notnull()]
    df = df[df['userId'].notnull()]

    data_preprocessing = dataUtils.DataUtils()

    url_list = df['url'].tolist()

    df['url'] = data_preprocessing.filter_data(url_list)


    df['userId'] = pd.factorize(df['userId'])[0]
    df['documentId'] = pd.factorize(df['documentId'])[0]

    df['time'] = converte_seconds_to_date_format(df, column='time')
    df['publishtime'] = converte_seconds_to_date_format(df, column='publishtime')

    return df

def find_documents_in_common(userId_1, userId_2):
    '''
        Input: two userIds to look up the users in the dataframe. 
        Return: a list of documentsIds that the user1 and user2 have in common, None is exluded from the list.
    '''
    user1_documentids = df.loc[df.userId == userId_1, ['documentId']].documentId.dropna().unique()
    user2_documentids = df.loc[df.userId == userId_2, ['documentId']].documentId.dropna().unique()
    

    documents_in_common_user1_user2 = [ x for x, y in zip(user1_documentids, user2_documentids) if x == y ]
    #print("user1: \n", user1_documentids)
    #print("user2: \n", user2_documentids)
    #print("documents in common: \n", documents_in_common_user1_user2)
    
    return documents_in_common_user1_user2

def cosine_similiarity(df):
    '''
        Loop through activeTimes and find similarities for one user towards every users in the system
        sim(A, B) = Ai * Bi / sqrt( Ai ^2 ) * sqrt( Bi ^2 )
    '''
    similarity_dataFrame = pd.DataFrame(columns=df['userId'])
    users_unique = df['userId'].unique()
    nr_users = len(users_unique)

    sim_matrix = np.zeros((nr_users, nr_users), float)
    print("similarity..")
    for userId_1 in users_unique:
        for userId_2 in users_unique:
            if userId_1 == userId_2:
                pass #same user so skip
            elif find_documents_in_common(userId_1, userId_2):

                user1_table = df.loc[df.userId == userId_1, ['activeTime','documentId']]
                user2_table = df.loc[df.userId == userId_2, ['activeTime', 'documentId']]

                # User1's activeTime1 * User2's activeTime1 aka "dot product" of user1 and user2
                multiply_user1_user2 = [x*y for x, y, in zip(user1_table['activeTime'], user2_table['activeTime'])]
                multiply_user1_user2 = sum(multiply_user1_user2)

                # we need to check here incase sum(activities) will yield 0 allthough they are not wrong/missing.
                # e.g. activeTime = 10 and activeTime = -10 will yield sum() == 0
                if multiply_user1_user2 == 0:
                    #print("user1_table: \n", user1_table)
                    #print("user1_table: \n", user2_table)
                    #print("welcome")
                    continue

                multiply_user1_user2 = round(multiply_user1_user2, 2)

                # user1's activeTime1^2  
                user1_square_sum = round(sum( [np.square(x) for x in user1_table['activeTime'] if x != 0.0] ), 2) 
                
                user2_square_sum = round(sum( [np.square(x) for x in user2_table['activeTime'] if x != 0.0] ), 2) 

                
                #check that multiply_user1_user2 are actually a int list and gets a correct sum
                                #sum dot product        /   squareRoot(user1's activeTime^2) * quareRoot(user2's activeTime^2)
                temp_user1 = round(np.sqrt(user1_square_sum), 4)
                temp_user2 = round(np.sqrt(user2_square_sum), 4)
                temp_user1_w_user2 = temp_user1 * temp_user2

                cosine_sim = round((multiply_user1_user2 / temp_user1_w_user2), 4)

                sim_matrix[ int(userId_1) ] [ int(userId_2) ] += cosine_sim


    return sim_matrix
                
def converte_seconds_to_date_format(df, column):
    '''
        Convert either unix-time from number of seconds since 1970 to a datetime format.
        The function will also convert datetime format to not include timezone, and be formatted correctly.
        Currently works on: publishtime and time. df must be a dataframe. column must be a string.
        Example:
        input: 1483225227
        output: 2016-12-31 23:00:27
        -------------------------------
        input: 2017-01-01T16:51:55.000Z
        output: 2017-01-01 16:51:55
    '''
    if column == 'time':
        try:
            return pd.to_datetime(df[str(column)], unit='s')
        except:
            print("Datetime conversion of seconds to date format failed..")
    elif column == 'publishtime':
        try: 
            return pd.to_datetime(df[str(column)]).dt.tz_localize(None)
        except:
            print("Datetime conversion date to correct date format failed..")
    else:
        return np.nan

def print_statistics(df):
    total_documentIds = df.documentId.value_counts(dropna=False).sum()
    documentIds_None =  df.documentId.isna().sum()

    total_frontpages =  df[df.url == "http://adressa.no"].documentId.value_counts(dropna=False).sum() # total frontpages with the ones that have documentId = None
    
    documentId_NoneNOT_front = documentIds_None - total_frontpages 

    print(
        "total documentIds in the system registered, aka events: ", total_documentIds, "\n",
        "events that consist of documentIds that are None: ", documentIds_None, "\n",

        "documents that are frontpage: ", total_frontpages, "\n",

        "documentId which are not frontpage and None: ", documentId_NoneNOT_front, "\n",
        
        "total unique documents in the system: ", str(len(df['documentId'].unique()) - 1), "\n", # - 1 because of None
    )

def save_matrix_to_file_numpy_txt(matrix):
    with open('sim_matrix.txt') as f:
        for line in matrix:
            np.savetxt(f, line, fmt='%.2f')

def save_matrix_to_file_pandas_csv(matrix):
    df = pd.DataFrame(data=matrix.astype(float))
    df.to_csv('outfile.csv', sep=' ', header=False, float_format='%.2f', index=False)

def load_matrix_from_file_numpy_txt(filename, number_of_columns):
    matrix = np.loadtxt(str(filename), usecols=range(number_of_columns))
    return matrix


if __name__ == '__main__':
    df = load_data("active1000")
    #df = load_one_file('active1000/20170101')
    #new_df = pd.read_csv('df_with_average_activetime.csv')
    #df = pd.read_csv('df_with_average_activetime.csv')

    print_statistics(df)    
    print(df.head())
    df = preprocessing_data(df)
    print_statistics(df)    
    print(df.head())

    print("number of activeTimes that are none: ", df.loc[df.activeTime.isnull(), 'activeTime'])
    #print("number of activeTimes that are none: ", df.loc[df.activeTime.isnull(), 'activeTime']).value_counts(dropna=False)

    start = timer()
    df = replace_none_activetime_with_average(df)
    end = timer()
    print("The time that went by for replace None with average: ", end-start, "seconds")

    start = timer()
    df = centered_cosine(df)
    end = timer()
    print("The time that went by for centered_cosine: ", end-start, "seconds")

    start = timer()
    sim_matrix = cosine_similiarity(df)
    end = timer()
    print("The time that went by for similarity comparison: ", end-start, "seconds")
    
    save_matrix_to_file_numpy_txt(sim_matrix)
    
    #load_matrix_from_file_numpy_txt('sim_matrix.txt', 3)

    print(df.head())
    
    #TODO:
    # We want to find a Neighborhood aka a set of users who are most similar to target user
    # Everyone in that neighborhood must have rated the same documentId that we are considering for the target user
    # - Let Rx be a vector of user X's ratings
    # - Let N be the set of k users most similar to X, who have also rated item i
    # - Option 1: take an average of the neighborhoods rating of item i and that will be the rating for X to item i.
    # - Option 2: Weighted average of neighboorhood, 







    '''

    **************************************************************************************************

                #cosine_sim_list.append(str(cosine_sim))
                #userId_2_list.append(str(userId_2)) 
                    
                #df['cosineSim_activeTime'] =  str(cosine_sim) + "|" + str(userId_1) + " compared to " + str(userId_2)

            else: # They are not sharing any documents in common, therefore they are similar with 0.
                
                #cosine_sim_list.append(str(0))
                #userId_2_list.append(str(userId_2))    

        #TODO: Build the dataframe
        #TODO: need to find another way






        #similarity_dataFrame.insert(loc=len(similarity_dataFrame.index))

        similarity_dataFrame.insert(loc=len(similarity_dataFrame.index), column=str(userId_1), value=cosine_sim_list)
        print(
            similarity_dataFrame.head()
        )
        similarity_dataFrame.insert(loc=len(similarity_dataFrame.index), column=str("User: " + str(userId_1) + "compared_to"), value=userId_2_list)

        print(
            similarity_dataFrame.head()
        )




    ********************************************************************************************************************************************

        #dataUtils = dataUtils.DataUtils()
    #print(df.head())   

    #print(df.documentId.tail())
    #print(df.documentId.tail().isna())

    #print(
        #"df.documentId.valueCounts", df.documentId.value_counts().sum(), "\n",
        #"precante of none documentIds: \n", df['documentId'].isna().mean()
    #)
    
    tempy = 123
    all_documents = df.documentId.value_counts(dropna=False)
    print(
       "Value 1: \n", df.documentId.value_counts(dropna=False), "\n",
       "**************************************************************************** \n",
        "value 2: \n", df.groupby(df.url == "http://adressa.no").documentId.value_counts(dropna=False, normalize=True), "\n"
       "**************************************************************************** \n",
        "value 3: \n", df.loc[ df.url == "http://adressa.no"].documentId.value_counts(dropna=False)
    )
    
    print(
        "**********************************************************************"
    )
    

    print( 
        #df.loc[df.userId == 'cx:10k2wzm1b3jsk2y5ym1utnjo97:2kefzqaxe9jx7', ['userId','activeTime', 'documentId']]
    )

    #df['true_docId'] =  
    # df.loc[df.userId == 'cx:10k2wzm1b3jsk2y5ym1utnjo97:2kefzqaxe9jx7', ['userId','activeTime', 'documentId']].documentId == 'NaN'
    print(
        #df.loc[df.userId == 'cx:10k2wzm1b3jsk2y5ym1utnjo97:2kefzqaxe9jx7', ['userId','activeTime', 'documentId']]
    )

    
    print(
        df.loc[df.userId == 'cx:10k2wzm1b3jsk2y5ym1utnjo97:2kefzqaxe9jx7', ['userId','activeTime']].documentId == 'NaN'

    )
    
    '''