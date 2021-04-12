from dataAggregator import DataAggregator
from datetime import time
import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
import dataUtils

from numpy import dot, nan
from numpy.linalg import norm
from timeit import default_timer as timer
#from pandas._libs.tslibs import Timestamp
#from scipy import spatial
#from pandas.tseries.offsets import Second
#from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine_similarity
from sklearn.model_selection import train_test_split as sk_train_test_split



class UserBasedRecommender:
    '''
        Class that house all the functionality related to a user-basec recommender
        the class holds onto all its variables, dataframes, matrixes and list so the class
        is probably quite large object, but that how it ended up being anyways.
        Remember to run start_recommender() to actually start anything.
    '''

    def __init__(self) -> None:
        # data processing
        self.data_processing_class = 0
        self.df = 0
        self.events = 0
        self.categories = 0

        # Data generating
        self.aggregator = 0

        self.train = 0
        self.test = 0
        self.nr_tests = 0
        self.nr_train = 0
        self.nr_users = 0
        self.nr_articles = 0
        self.split_time = 0
        self.split_eventId = 0
        self.nr_train_articles = 0
        self.first_relevant_article = 0
        self.nr_test_articles = 0

        # Data generate rating and sim matrix's
        self.rating_matrix = np.zeros((self.nr_users, self.nr_articles))
        self.sim_matrix = np.zeros((self.nr_users, self.nr_users))
        self.sim_pearson_correlation = np.zeros((self.nr_users, self.nr_users))
    
    def start_recommender(self, number_of_files_loaded):

        self.data_processing(number_of_files_loaded)
        self.data_generating()
        self.data_generate_rating_sim_matrix()

    def new_replace_none_w_average(self, df):
        '''
            replacing ac = None with that avg activeTime in the dataset of eventIds with 
            same documentId and a valid do
             divided on total events
        '''
        documentIds_null = df.loc[df.activeTime.isnull(), ['documentId','activeTime']]

        for _, row  in documentIds_null.iterrows():
            #documentId = row[0] #activeTime = row[1] #userId = row[2]
            # go to the agregator class that hold the previously generated "helper data" acess correct column.
            if np.isnan(row[1]):
                documentIds_null.loc[documentIds_null.documentId == row[0], 'activeTime'] = self.aggregator.articles.at[row[0], "averageActiveTime"]
            
        df.loc[df.activeTime.isnull(), ['documentId','activeTime']] = documentIds_null
    
        return df
        
    def converte_seconds_to_date_format(self, df, column):
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
            return 0

    def data_processing(self, number_of_files_loaded):
        print("starting data processing...")
        self.data_processing_class = dataUtils.DataUtils()
        # Load data as a list
        list_of_data = self.data_processing_class.load_data("active1000", num=number_of_files_loaded)
        # Filter the data
        list_of_filtered_data = self.data_processing_class.filter_data(list_of_data)
        # Inedx data with ints, docuemntId and userId
        list_of_indexed_data = self.data_processing_class.index_data(list_of_filtered_data)
        # Returns a DataFrame based on the list
        self.df = self.data_processing_class.get_dataframe(list_of_indexed_data)
        # fix dates 
        #self.df['time'] = self.converte_seconds_to_date_format(self.df, column='time')
        #self.df['publishtime'] = self.converte_seconds_to_date_format(self.df, column='publishtime')
        # Clean up categorys and fix dates again
        self.events, self.categories = self.data_processing_class.process_data(self.df)
        print("done with data processing")

    def data_generating(self):
        '''
            Generate helping data, or more data to have the calculations before needing them.
            E.g. avrage-activeTime of a userId or a documentId.
        '''
        print("starting data generating...")

        #Generate train-test split, to get helping data for the dataset.
        self.train, self.test = sk_train_test_split(self.events, test_size=0.2, shuffle=False)
        self.nr_tests = len(self.test.index)
        self.nr_train = len(self.train.index)
        self.nr_users = self.data_processing_class.nextUserID
        self.nr_articles = self.data_processing_class.nextDocumentID

        # want to find the time according to the dataset where the split between train and test
        self.split_time = self.train.iloc[-1]["eventTime"]    # was performed.
        self.split_eventId = self.train.iloc[-1]["eventId"]
        self.nr_train_articles = self.train['documentId'].max()
        self.first_relevant_article = max( self.test['documentId'].min(), int( self.nr_train_articles * 0.95))
        # Cut most of the articles, and do the assumption that the last 5% articles are 
        # the ones of relevance to us. This is the last 5 % of the train-set talking about.
        self.nr_test_articles = self.nr_articles - self.first_relevant_article

        # Sending in train set to aggregator.
        self.aggregator = DataAggregator(categories=self.categories, strict=False)

        # multithread to speed up processing.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.aggregator.generateArticleData, self.train, self.nr_articles)
            executor.submit(self.aggregator.generateUserData, self.train, self.nr_users)
        
        self.train = self.data_processing_class.fill_missing( self.train, self.aggregator.articles, self.aggregator.users)
        print("done with data generating.")

    def data_generate_rating_sim_matrix(self):
        print("starting data generate_sim_rating...")

        self.df = pd.DataFrame(self.train)

        self.df = self.new_replace_none_w_average(self.df)

        df = cosine_mean(self.df)

        self.aggregator.generateRatingMatrix(self.train, self.nr_users, self.nr_articles)

        self.rating_matrix = self.aggregator.ratings
        
        self.sim_matrix = self.new_cosine_similarity(self.rating_matrix, self.df)
        self.sim_pearson_correlation = self.new_cosine_pearson_similarity(self.rating_matrix, self.df)

        print("done with data generate_sim_rating.")

    def user_based_predictions(self, userId, documentId, time=None):
        user_1 = userId
        users_that_are_similar_to_user_1 = self.sim_matrix[userId].max()



        return print("yo")
    
    def create_rating_matrix(self, df):
        '''
            Map activeTime per userId and documentId
            x-axis = documentId
            y-axis = userId
            position(x)(y) = activetime
        '''
        rating_matrix = np.zeros((self.nr_users,self.nr_articles))
        for _, event in df.iterrows():
            rating_matrix[ int( event['userId'] ), int( event['documentId'] ) ] = event['activeTime']

        return rating_matrix

    def new_cosine_similarity(self, rating_matrix, df):
        '''
            Previosly method was to slow..
            dot product of 2 arrays divided on absolute values of each array then
            multiplied together results in a cosine_sim
        '''
        users_unique = df['userId'].unique()
        nr_users = len(users_unique)
        sim_matrix = np.zeros((nr_users, nr_users), float)
        
        for userId_1 in users_unique:
            for userId_2 in users_unique:

                array_1 = rating_matrix[userId_1]
                array_2 = rating_matrix[userId_2]

                sim_matrix[ int(userId_1) ] [ int(userId_2) ] = np.dot(array_1, array_2) / (norm(array_1) * norm(array_2))
                    
        return sim_matrix

    def new_cosine_pearson_similarity(self, rating_matrix, df):
        '''
            Again my old def was to slow..
            sane as cosine_sim, allthough the pearson correlation takes into consideration
            the avrage mean, and mean-center the values of each array before doing 
            cosine similarity.
        '''
        users_unique = df['userId'].unique()
        nr_users = len(users_unique)
        sim_matrix = np.zeros((nr_users, nr_users), float)
        
        for userId_1 in users_unique:
            for userId_2 in users_unique:

                array_1 = rating_matrix[userId_1]
                array_2 = rating_matrix[userId_2]    

                sim_matrix[ int(userId_1) ] [ int(userId_2) ] = np.corrcoef(array_1, array_2)[0][1]
                    
        return sim_matrix

    def cosine_mean(self, df):

        users_unique = df['userId'].unique()

        print(df.head)
        for userId in users_unique:
            activeTimeS = df.loc[df.userId == userId, ['activeTime']]
            #activeTimes_numeric = sum(activeTimeS['activeTime'].tolist())
            user_average = sum(activeTimeS.astype(int)) / len(activeTimeS['activeTime'])
            #user_average_2 = aggregator.users.at[userId, 'averageViewTime']
            if np.isnan(user_average):
                print("wtf...... whyt?")
            temp = []
            for _, row in activeTimeS.iterrows():
                # activetime = row[0]
                #temp.append(int(row[0] - user_average))
                row[0] = (int(row[0]) - user_average)

            df.loc[df.userId == userId, ['activeTime']] = activeTimeS
        print(df.head)
        
        return df

if __name__ == '__main__':
    start = timer()
    UsB_recommender = UserBasedRecommender()
    UsB_recommender.start_recommender(number_of_files_loaded=1)
    
    test = UsB_recommender.test
    for index, row in test.iterrows():
        print(row)

    UsB_recommender.user_based_predictions( )
    end = timer()
    print("The time that went by for similarity comparison: ", end-start, "seconds")
    print("we made it?")

    #TODO:
    # We want to find a Neighborhood aka a set of users who are most similar to target user
    # Everyone in that neighborhood must have rated the same documentId that we are considering for the target user
    # - Let Rx be a vector of user X's ratings
    # - Let N be the set of k users most similar to X, who have also rated item i
    # - Option 1: take an average of the neighborhoods rating of item i and that will be the rating for X to item i.
    # - Option 2: Weighted average of neighboorhood, 


    '''
    def load_data(path):
    
    #    Load events from files and convert to dataframe.
    
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
    
    #    Less loading time with working on smaller dataset
    #    Will always load file 20170101 or a specific file given by
    #    path. Load events from files and convert to dataframe.
    
    map_lst=[]
    file_name=os.path.join(path)
    if os.path.isfile(file_name):
        for line in open(file_name):
            obj = json.loads(line.strip())
            if not obj is None:
                map_lst.append(obj)
    return pd.DataFrame(map_lst) 

def train_test_split(ratings, fraction=0.2):
    # Leave out a fraction of dataset for test use
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
    
        #Basically removing 2/3 of the dataset....
    

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
    
    #    Centered Cosine
    #    1. We take the mean of user's activeTime.
    #    2. Subtract that mean from all individual activeTime divided by the total number of activeTime by the user.
    #    3. For all the activeTimes where there is no time(NaN), we replace it with 0.
    
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
     
    #    Finding the documentIds with None activeTime. Then find if there exist others in the system
    #    that have looked at those specific documentIds with a valid activeTime. Replace the none activeTime
    #    with average time of activeTime found.

    
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
        #avg = int(document_list_sum / document_list_len)

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

    df['userId'] = pd.factorize(df['userId'])[0]
    df['documentId'] = pd.factorize(df['documentId'])[0]

    df['time'] = converte_seconds_to_date_format(df, column='time')
    df['publishtime'] = converte_seconds_to_date_format(df, column='publishtime')

    return df

def find_documents_in_common(userId_1, userId_2):
    
    #    Input: two userIds to look up the users in the dataframe. 
    #    Return: a list of documentsIds that the user1 and user2 have in common, None is exluded from the list.
    
    user1_documentids = df.loc[df.userId == userId_1, ['documentId']].documentId.dropna().unique()
    user2_documentids = df.loc[df.userId == userId_2, ['documentId']].documentId.dropna().unique()
    

    documents_in_common_user1_user2 = [ x for x, y in zip(user1_documentids, user2_documentids) if x == y ]
    #print("user1: \n", user1_documentids)
    #print("user2: \n", user2_documentids)
    #print("documents in common: \n", documents_in_common_user1_user2)
    
    return documents_in_common_user1_user2

def cosine_similarity(df):
    
    #    Loop through activeTimes and find similarities for one user towards every users in the system
    #    sim(A, B) = Ai * Bi / sqrt( Ai ^2 ) * sqrt( Bi ^2 )
    

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
                
                user1_array = np.array(user1_table['activeTime'])
                user2_array = np.array(user2_table['activeTime'])
                
                new_sim = np.dot(user1_array, user2_array)/(norm(user1_array)*norm(user2_array))

                new_sim_2 = 1 - spatial.distance.cosine(user1_array, user2_array)

                # User1's activeTime1 * User2's activeTime1 aka "dot product" of user1 and user2
                multiply_user1_user2 = [x*y for x, y, in zip(user1_table['activeTime'], user2_table['activeTime'])]
                multiply_user1_user2 = sum(multiply_user1_user2)

                # we need to check here incase sum(activities) will yield 0 allthough they are not wrong/missing.
                # e.g. activeTime = 10 and activeTime = -10 will yield sum() == 0
                if multiply_user1_user2 == 0:
                    
                    continue

                multiply_user1_user2 = round(multiply_user1_user2, 2)

                # user1's activeTime1^2  
                user1_square_sum = round(sum( [np.square(x) for x in user1_table['activeTime'] if x != 0.0] ), 2)
                #new_array_1 = user1_array@user2_array.T
                #print(new_array_1) 
                
                user2_square_sum = round(sum( [np.square(x) for x in user2_table['activeTime'] if x != 0.0] ), 2) 
                #new_array_2 = user2_array@user1_array.T
                #print(new_array_2)
                
                #check that multiply_user1_user2 are actually a int list and gets a correct sum
                                #sum dot product        /   squareRoot(user1's activeTime^2) * quareRoot(user2's activeTime^2)
                temp_user1 = round(np.sqrt(user1_square_sum), 4)
                temp_user2 = round(np.sqrt(user2_square_sum), 4)
                temp_user1_w_user2 = temp_user1 * temp_user2


                cosine_sim = round((multiply_user1_user2 / temp_user1_w_user2), 4)

                sim_matrix[ int(userId_1) ] [ int(userId_2) ] += cosine_sim

    return sim_matrix
                
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

def create_rating_matrix(df):
    nr_users = len(df['userId'].unique())
    nr_documents = len(df['documentId'].unique())

    rating_matrix = np.zeros((nr_users,nr_documents))
    for _, event in df.iterrows():
        rating_matrix[ int( event['userId'] ), int( event['documentId'] ) ] = event['activeTime']

    return rating_matrix

def new_cosine_similarity(rating_matrix, df):
    users_unique = df['userId'].unique()
    nr_users = len(users_unique)
    sim_matrix = np.zeros((nr_users, nr_users), float)
    
    for userId_1 in users_unique:
        for userId_2 in users_unique:
            #if userId_1 == userId_2:
            #    pass #same user so skip
            #elif find_documents_in_common(userId_1, userId_2):
            array_1 = rating_matrix[userId_1]
            array_2 = rating_matrix[userId_2]

            sim_matrix[ int(userId_1) ] [ int(userId_2) ] = np.dot(array_1, array_2) / (norm(array_1) * norm(array_2))
                
    return sim_matrix

def new_cosine_pearson_similarity(rating_matrix, df):
    users_unique = df['userId'].unique()
    nr_users = len(users_unique)
    sim_matrix = np.zeros((nr_users, nr_users), float)
    
    for userId_1 in users_unique:
        for userId_2 in users_unique:
            #if userId_1 == userId_2:
            #    pass #same user so skip
            #elif find_documents_in_common(userId_1, userId_2):
            array_1 = rating_matrix[userId_1]
            array_2 = rating_matrix[userId_2]
            
            #new_sim = np.corrcoef(array_1, array_2)[0][1]

            sim_matrix[ int(userId_1) ] [ int(userId_2) ] = np.corrcoef(array_1, array_2)[0][1]

            #sim_matrix[ int(userId_1) ] [ int(userId_2) ] = np.dot(array_1, array_2) / (norm(array_1) * norm(array_2))
                
    return sim_matrix

def cosine_mean(df):
    users_unique = df['userId'].unique()

    print(df.head)
    for userId in users_unique:
        activeTimeS = df.loc[df.userId == userId, ['activeTime']]
        #activeTimes_numeric = sum(activeTimeS['activeTime'].tolist())
        user_average = sum(activeTimeS.astype(int)) / len(activeTimeS['activeTime'])
        #user_average_2 = aggregator.users.at[userId, 'averageViewTime']
        if np.isnan(user_average):
            print("wtf...... whyt?")
        temp = []
        for _, row in activeTimeS.iterrows():
            # activetime = row[0]
           #temp.append(int(row[0] - user_average))
           row[0] = (int(row[0]) - user_average)

        df.loc[df.userId == userId, ['activeTime']] = activeTimeS
    print(df.head)
    return df

    
    df = pd.DataFrame(train)

    df = new_replace_none_w_average(df)

    #df = cosine_mean(df)

    aggregator.generateRatingMatrix(train, nr_users, nr_articles)

    user_based_recommender = UserBasedRecommender()
    

    rating_matrix = aggregator.ratings
    
    sim_matrix = new_cosine_similarity(rating_matrix, df)
    sim_pearson_correlation = new_cosine_pearson_similarity(rating_matrix, df)

    
    user_based_predictions(test, sim_matrix, rating_matrix, df)

   

    print(df.head())

    print("************************** Hello ***************************")





    
    print(sim_matrix[0:,])


    
    print_statistics(df)    
    print(df.head())
    #df = preprocessing_data(df)
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
    sim_matrix = cosine_similarity(df)
    end = timer()
    print("The time that went by for similarity comparison: ", end-start, "seconds")
    
    save_matrix_to_file_numpy_txt(sim_matrix)
    
    #load_matrix_from_file_numpy_txt('sim_matrix.txt', 3)

    print(df.head())


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