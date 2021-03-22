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

def remove_all_nulls(df):
    print(        df.shape    )
    df = df[df['eventId'].notnull()]
    df = df[df['category'].notnull()]
    df = df[df['activeTime'].notnull()]
    df = df[df['title'].notnull()]
    df = df[df['url'].notnull()]
    df = df[df['publishtime'].notnull()]
    df = df[df['time'].notnull()]
    df = df[df['documentId'].notnull()]
    df = df[df['userId'].notnull()]
    print(        df.shape    )

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
               activeTimeS.iloc[i] = float(activeTimeS.iloc[i] - mean_of_that_user)
            else: 
                activeTimeS.iloc[i] = 0.0         
        df.loc[df.userId == userId, ['activeTime']] = activeTimeS

    # SIMILIARITY CHECK --
    cosine_similiarity(df)


    return df

 
def cosine_similiarity(df):
    '''
        Loop through activeTimes and find similarities for one user towards every users in the system
        sim(A, B) = Ai * Bi / sqrt( Ai ^2 ) * sqrt( Bi ^2 )
    '''
    for userId_1 in df['userId']:
        #table consist of activeTime and documentId so we keep track of which documents we are comparing.
        user1_table = df.loc[df.userId == userId_1, ['activeTime','documentId']]

        for userId_2 in df['userId']: # The next user we want to loop on

            # iloc allows us to loop on the list
            user2_table = df.loc[df.userId == userId_2, ['activeTime', 'documentId']]
            
            if user1_table.equals(user2_table):
                pass #Compare users, if same, skip to next user.
            else: 
                # TODO:
                # need to do a check on documentIds
                # If they dont share documentId's then skip



                # User1's activeTime1 * User2's activeTime1 aka "dot product" of user1 and user2
                multiply_user1_user2 = [x*y for x, y, in zip(user1_table['activeTime'], user2_table['activeTime'])]

                # user1's activeTime1^2  
                user1_square_sum = sum( [np.square(x) for x in user1_table['activeTime'] if x != 0.0] ) 
                
                user2_square_sum = sum( [np.square(x) for x in user2_table['activeTime'] if x != 0.0] ) 
                
                #check that multiply_user1_user2 are actually a int list and gets a correct sum
                                #sum dot product        /   squareRoot(user1's activeTime^2) * quareRoot(user2's activeTime^2)
                cosine_sim = sum(multiply_user1_user2) / ( np.sqrt(user1_square_sum) * np.sqrt(user2_square_sum) )


                df['cosineSim_activeTime'] =  str(cosine_sim + "|" + str(userId_1) + " compared to " + str(userId_2))
                
        
   

def converte_seconds_to_date_format(df, column=False):
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
    try:
        return pd.to_datetime(df[str(column)], unit='s')
    except:
        print("Datetime conversion of seconds to date format failed..")
    
    try: 
        return pd.to_datetime(df[str(column)]).dt.tz_localize(None)
    except:
        print("Datetime conversion date to correct date format failed..")
    return np.nan


def suprise_CF(df):
    '''
        collaborative filtering user based with the suprise package from scikit.
    '''
    from surprise import Reader, Dataset, SVD, NMF
    from surprise.model_selection import cross_validate

    reader = Reader(rating_scale=(1.0, 899.0))

    df['activeTime'] = replace_NaN_w_0(df, 'activeTime')
    df['userId'] = replace_NaN_w_0(df, 'userId')
    #df['publishtime'] = replace_NaN_w_0(df, 'publishtime')
    df['time'] = replace_NaN_w_0(df, 'time')
    
    
    #print("Publishtime: \n", df['publishtime'].tail())
    #df['publishtime'] = converte_seconds_to_date_format(df, 'publishtime')
    #print("Publishtime: \n",df['publishtime'].tail())

    #print("Time: \n", df.time.head() )
    #df['time'] = converte_seconds_to_date_format(df, 'time')
    #print("Time: \n", df.time.head() )
   
    print("stop here")
    



    data = Dataset.load_from_df(df[['userId', 'activeTime', 'time']], reader)

    data.split(n_folds = 5)

    # Singular Value Decomposition (SVD)
    algorithm = SVD()
    cross_validate(algorithm, data, measure=['RMSE'], cv=5, verbose=True)



    # None negative Matrix Factorization
    #algorithm = NMF()
    #evaluate(algorithm, data, measure=['RMSE'])
    print("Now we are here")

if __name__ == '__main__':
    #df = load_data("active1000")
    df = load_one_file('active1000/20170101')

    dataUtils = dataUtils.DataUtils()
    

    activeTime = df.activeTime
    print(
        "this is activeTime: ", activeTime.max(), " _min: ",activeTime.min()
    )
    print(df.head())

    #suprise_CF(df)




    print(df.head())   
    start = timer()
    df = centered_cosine(df)
    end = timer()
    print("The time that went by: ", end-start, "seconds")
    print(df.head())











    '''
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