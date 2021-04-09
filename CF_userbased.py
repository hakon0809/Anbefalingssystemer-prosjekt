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
        Basically removing 66% of the dataset..
    '''
    amount_before = df.shape[0]

    df = df[df['eventId'].notnull()]
    
    df = df[df['documentId'].notnull()]
    df = df[df['userId'].notnull()]

    #df = df[df['category'].notnull()]
    #df = df[df['activeTime'].notnull()]
    #df = df[df['title'].notnull()]
    #df = df[df['url'].notnull()]
    #df = df[df['publishtime'].notnull()]
    #df = df[df['time'].notnull()]

    amount_after = df.shape[0]    

    amount_precantege_left_with = int((amount_after/amount_before)*100)
    amount_removed_precentage = 100 - amount_precantege_left_with
    print("The amount of the dataset that was discarded: ", amount_removed_precentage, "%")
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
        # Access each individual activeTime of a specific userId
        activeTimeS = df.loc[df.userId == userId, ['activeTime']]

        # Loop on all activitimes of the specific userId, updates its activeTime
        for i in range(len(activeTimeS)):
            if activeTimeS.iloc[i].notna().bool() == True: # NaN check
               activeTimeS.iloc[i] = float(activeTimeS.iloc[i] - round(mean_of_that_user, 2))
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
    #counter
    i = 0
    half_way = len(documentIds_null['documentId'])
    # Find all documentIds with activeTime == None/NaN/null/whatever
    for docuId in documentIds_null['documentId']:
        #Increment
        i +=0
        # Find all activeTimes of the first documentId==None. NB Here it is important that we find only real values
        # therefore the "..notnull()"
        new_documentIds_value = documentIds_value[documentIds_value.documentId == docuId].activeTime
        
        document_list_sum = int(new_documentIds_value.sum())
        document_list_len = int(len(new_documentIds_value))

        if document_list_len <= 0 | document_list_sum <= 0:
            pass
        else:
            documentIds_null.loc[documentIds_null.documentId == docuId, 'activeTime'] = int(document_list_sum / document_list_len)
        
        if half_way == i:
            None

    df.loc[df.activeTime.isnull(), ['documentId','activeTime']] = documentIds_null

    df.to_csv('df_with_average_activetime.csv')

    return df

def preprocessing_data(df):
    

    #df = remove_all_none_values(df)
    print(df.dtypes  )
    df = df[df['eventId'].notnull()]
    df = df[df['documentId'].notnull()]
    df = df[df['userId'].notnull()]

    df['userId'] = pd.factorize(df['userId'])[0]
    df['documentId'] = pd.factorize(df['documentId'])[0]

    df['time'] = converte_seconds_to_date_format(df, column='time')
    df['publishtime'] = converte_seconds_to_date_format(df, column='publishtime')


    #TODO: strings are treated as integers or visa versa
    #print(df['url'].head())
    #dataProcessing = dataUtils.DataUtils()
    #df['url'] = dataProcessing.filter_data(df)

    start = timer()
    df = replace_none_activetime_with_average(df)
    end = timer()
    print("The time that went by for replace none with average: ", end-start, "seconds")

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

    for userId_1 in df['userId'].unique():
        cosine_sim_list = []
        userId_2_list = []
        sim_dict = {}
        #table consist of activeTime and documentId so we keep track of which documents we are comparing.
        #user1_table = df.loc[df.userId == userId_1, ['activeTime','documentId']]

        for userId_2 in df['userId'].unique(): # The next user we want to loop on
            
            # iloc allows us to loop on the list
            #user2_table = df.loc[df.userId == userId_2, ['activeTime', 'documentId']]
        
            if userId_1 == userId_2:
                pass #same user so skip
            elif find_documents_in_common(userId_1, userId_2):
                common_docs = find_documents_in_common(userId_1, userId_2)
                user1_table = df.loc[df.userId == userId_1, ['activeTime','documentId']]
                user2_table = df.loc[df.userId == userId_2, ['activeTime', 'documentId']]
                #print("user1_table: \n", user1_table)
                #print("user2_table: \n", user2_table)

                
                #for docs in find_documents_in_common(userId_1, userId_2):
                #user1_table = df.loc[(df.userId == userId_1) & (df.documentId == docs), ['activeTime','documentId']]
                #user2_table = df.loc[(df.userId == userId_2) & (df.documentId == docs), ['activeTime','documentId']]

                # TODO:
                # need to do a check on documentIds
                # If they dont share documentId's then skip
                # activete = None --> average av alle som har sett på den artikkel.

                # User1's activeTime1 * User2's activeTime1 aka "dot product" of user1 and user2
                multiply_user1_user2 = [x*y for x, y, in zip(user1_table['activeTime'], user2_table['activeTime'])]
                multiply_user1_user2 = sum(multiply_user1_user2)
                multiply_user1_user2 = round(multiply_user1_user2, 2)
                # user1's activeTime1^2  
                user1_square_sum = round(sum( [np.square(x) for x in user1_table['activeTime'] if x != 0.0] ), 2) 
                
                user2_square_sum = round(sum( [np.square(x) for x in user2_table['activeTime'] if x != 0.0] ), 2) 
                
                #check that multiply_user1_user2 are actually a int list and gets a correct sum
                                #sum dot product        /   squareRoot(user1's activeTime^2) * quareRoot(user2's activeTime^2)
                temp_user1 = round(np.sqrt(user1_square_sum), 4)
                temp_user2 = round(np.sqrt(user2_square_sum), 4)
                temp_user1_w_user2 = temp_user1 * temp_user2
                cosine_sim_2 = round((multiply_user1_user2 / temp_user1_w_user2), 4)

                cosine_sim = round (multiply_user1_user2 /  round(np.sqrt(user1_square_sum), 4) * round( np.sqrt(user2_square_sum),4), 2)

                #cosine_sim_list.append({)

                cosine_sim_list.append(str(cosine_sim_2))
                userId_2_list.append(str(userId_2)) 
                    
                df['cosineSim_activeTime'] =  str(cosine_sim_2) + "|" + str(userId_1) + " compared to " + str(userId_2)

            else: # They are not sharing any documents in common, therefore they are similar with 0.
                cosine_sim_list.append(str(0))
                userId_2_list.append(str(userId_2))    

        #TODO: Build the dataframe
        #TODO: need to find another way

        #similarity_dataFrame.insert(loc=len(similarity_dataFrame.index))
        
        similarity_dataFrame.insert(loc=len(similarity_dataFrame.index), column=str(userId_1), value=cosine_sim_list)
        print(
            similarity_dataFrame.head()
        )
        similarity_dataFrame.insert(loc=len(similarity_dataFrame.index), column=str("User: " + str(userId_1) + "compared_to"), value=userId_2_list)


        #for i in len(userId_2_list):
        #    similarity_dataFrame.loc[i]

        #similarity_dataFrame.assign()
        #similarity_dataFrame = pd.DataFrame(cosine_sim_list, columns=userId_2_list, index=str(userId_1))

        #similarity_dataFrame[str("user: " + str(userId_1) + " similarity")] = cosine_sim_list
        #similarity_dataFrame[str(userId_1)] = cosine_sim_list
        #similarity_dataFrame[str(str(userId_1) + "_users_compared_to")] = userId_2_list
        #similarity_dataFrame.insert(loc=len(similarity_dataFrame.columns), column=str(userId_1), value=cosine_sim_list)
        #similarity_dataFrame.insert(loc=len(similarity_dataFrame.columns), column=str(userId_1 + " users"), value=userId_2_list)
        print(
            similarity_dataFrame.head()
        )
            
    return df, similarity_dataFrame
                
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

if __name__ == '__main__':
    df = load_data("active1000")
    #df = load_one_file('active1000/20170101')
    #df = pd.read_csv('df_with_average_activetime.csv')

    # TODO:
    # Focus on category after activetime
    # er ikke nødvendig å se på atributter av artikkler, når de skal



    print_statistics(df)    
    print(df.head())

    df = preprocessing_data(df)


    start = timer()
    df = centered_cosine(df)
    end = timer()
    print("The time that went by for centered_cosine: ", end-start, "seconds")

    # SIMILIARITY CHECK --
    start = timer()
    df, similarity_dataFrame = cosine_similiarity(df)
    end = timer()
    
    print("The time that went by for similarity comparison: ", end-start, "seconds")
    print(df.head())
    print("**********************************")
    print(similarity_dataFrame.head() )








    '''

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