import sys, os 
import csv 
import re
import sqlite3 
import logging 
import traceback 
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
from pprint import pprint 
from sklearn.decomposition import PCA 
from sklearn import preprocessing 
from datetime import datetime 
from gensim.models import word2vec
from sklearn.preprocessing import StandardScaler
from pickle import dump, load 
# NLP 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')

logging.getLogger().setLevel(logging.INFO)

class Database: 

    def __init__(self, fileDB):
        self.fileDB = fileDB 
        self.conn = self.create_connection() 
        self.model = word2vec.Word2Vec.load("model_gensim/word2vec.model")


    def create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.fileDB)
        except Exception as e:
            logging.warning(f"Error creating Connection to Database {self.fileDB}: Exception: {e}")

        return conn 

    def list_tables(self):
        """ 
            Lists all tables found in the input database 
        
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [x[0] for x in cursor.fetchall() if x[0] != 'summary']
    
    def get_columns(self, table_name):
        """ 
            Given a tablename as input, get_columns will returns the names of all columns inside this input table.
            The output here is list of tupels, where each tupel has a empty first index 
        """
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        return cursor.fetchall()

    def get_columns_names(self, table_name):
        """ 
            Given a tablename as input, get_columns will returns the names of all columns inside this input table.
            The output here is list strings.
        """
        cursor = self.conn.cursor()
         
        cursor.execute(f"PRAGMA table_info({table_name});")
        res = cursor.fetchall() 
        return [x[1] for x in res]

    def create_table_mapping(self):
        """ 
            Gives for every table the name of columns in it and returns it as dictionary 
        """
        res = {}
        all_tables = self.list_tables()
        for table in all_tables:
            
            res[table] = self.get_columns_names(table)
        return res 
    
    def read_csv(self, path):

        """ 
            Reads a csv file given as input 
        
        """
        logging.info(f"|-> Reading path {path}")
        data = pd.read_csv(path) 
        logging.info(f"|-> Successfully read data of shape {data.shape}")        
        return data
    
    def export_csv(self, data, name):
        """ 
            Exports a DataFrame object given as input data as csv file. The name of the file is then specified by the input argument name. 
        
        """
        logging.info(f"|-> Exporting Data to csv, filename : {name}")
        data.to_csv(name, index=False)
        logging.info(f"|-> Done writing data to csv, filename : {name}")
    
    def embed_sentence(self, sent):
        """ 
            convert sentence given as input argument sent into a embedding vector. All Stopword, special chars and NaNs are then 
            deleted from the representation. 
            
            Returns a list of words that are important for the meaning of the sentence 
        
        """
        if isinstance(sent, float):
            return []  # because there is entries without a charge (1300+ )
        else:

            porter = PorterStemmer()
            sent = sent.replace('/', ' ') 
            # convert to lower and tokenize 
            tokenized_sent = word_tokenize(sent.lower())
            # remove special characters 
            # remove punctuations/ symbols / numbers 
            tokenized_sent = [re.sub('[\W\_]','', word) for word in tokenized_sent if not word in stopwords.words('english')]
            # remove Nans and empty strings 
            tokenized_sent = [word for word in tokenized_sent if len(word) > 1]
            return tokenized_sent 

    def calculate_value(self, tokenized_charge):
        """ 
            This method becomes a list of tokens(important words) and converts this tokenized list into 
            a representation value used as a value for the charge for training. 
        """
        if not tokenized_charge:
            return -1.0
        else:

            to_multiply = []
            for word in tokenized_charge:
                vector_of_word = self.model.wv[word]
                to_multiply.append(vector_of_word)
                #logging.info("|-> Converted word <{word}> to vector and multiplying ")
            #logging.info("|-> Result of multiplication ")
            summation = np.add.reduce(to_multiply)
            summation = round(sum(summation) / summation.shape[0], 4)
            return summation 

    def embed_sentences(self, data, col):
        ''' 
            step 1) train a word2vec model on all possible words in charge column 
            step 2) convert all possible words in each row in data[col] to a vector using that model 
            step 3) summ all vectors of each word and then calculate the average 
            step 5) save the average as a sentence representation of the charge 

        '''

        corpus1 = data[col].values.astype('U').tolist()
        logging.info('|-> NLP preprocessing, this will take a while')
        corpus = [self.embed_sentence(sentence) for sentence in tqdm(corpus1)]
        logging.info('|-> NLP preprocessing is done ')
        
        # self.model = word2vec.Word2Vec(corpus, min_count=1)

        #logging.info('|-> Saving Word2Vec model.... ')
        #self.model.save("word2vec.model")
        #logging.info('|-> Saving Word2Vec model done' )
        
        # model = word2vec.Word2Vec.load("word2vec.model") # this is how you load word2vec model
        corpus_sent2vec = []
        
        for tokenized_charge in tqdm(corpus):
            #logging.info(f"|-> Working on word : {tokenized_charge}")
            summation = -1.0
            to_multiply = []
            if not tokenized_charge:
                corpus_sent2vec.append(summation)
            else:

                for word in tokenized_charge:
                    vector_of_word = self.model.wv[word]
                    to_multiply.append(vector_of_word)
                    #logging.info("|-> Converted word <{word}> to vector and multiplying ")
                #logging.info("|-> Result of multiplication ")
                summation = np.add.reduce(to_multiply)
                summation = round(sum(summation) / summation.shape[0], 4)
                corpus_sent2vec.append(summation)
                 
        data[col] = corpus_sent2vec
        return data 

    def categorize(self, data, cols):
        """ 
            This method creates categories for all columns in data specified by the list of strings cols. 
            
            Ex: for the column sex there is only two classes 1 for male and 0 for female.
        """
        logging.info("|-> Starting to categorize ")
        categories = {}
        for col in cols:
            logging.info(f"|-> categorizing {col}")
            data[col] = data[col].astype('category')
            categories[col] = dict(enumerate(data[col].cat.categories))
            data[col] = data[col].cat.codes
        logging.info("|-> Done categorizing ")
        return data, categories   
    
    def calculate_custody(self, data):
        """
            This method calculates calculates the custody period from the given custody_in and out. And does the same 
            thig for the jail period. Finally it saves the period it in a new column
        """ 
        logging.info('|-> Converting jail custody to datetime')
        data['in_custody']     = pd.to_datetime(data['in_custody'])
        data['out_custody']    = pd.to_datetime(data['out_custody'])
        logging.info('|-> Converting jail custody to datetime finished ')
        logging.info('|-> Subtracting datetimes for each entry ')
        data['jail_days'] = data['out_custody'].sub(data['in_custody']).dt.days
        
        # if the difference is only hours and the datetimes has same %Y%m%d then -1 retuned 
        # this line below will remove -1 and replace them with 0
        data['jail_days'] = data['jail_days'].apply(lambda x: 0 if x==-1 else x)
        
        logging.info('|-> Subtracting datetimes for each entry finished ')
        logging.info('|-> Deleting in_custody and out_custody')
        del data['in_custody']
        del data['out_custody'] 
        
        logging.info('|-> Converting prison custody to datetime')
        data['prison_in']     = pd.to_datetime(data['prison_in'])
        data['prison_out']    = pd.to_datetime(data['prison_out'])
        logging.info('|-> Converting prison custody to datetime finished ')
        logging.info('|-> Subtracting datetimes for each entry ')
        data['prison_days'] = data['prison_out'].sub(data['prison_in']).dt.days

        # if the difference is only hours and the datetimes has same %Y%m%d then -1 retuned 
        # this line below will remove -1 and replace them with 0
        data['prison_days'] = data['prison_days'].apply(lambda x: 0 if x==-1 else x)
        
        logging.info('|-> Subtracting datetimes for each entry finished ')
        logging.info('|-> Deleting in_custody and out_custody')
        del data['prison_in']
        del data['prison_out'] 

        logging.info('|-> Converting person c_jail to datetime')
        data['c_jail_in']   = pd.to_datetime(data['c_jail_in']) 
        data['c_jail_out']  = pd.to_datetime(data['c_jail_out']) 
        logging.info('|-> Converting person c_jail to datetime finished')
        logging.info('|-> Subtracting datetimes for each entry')
        data['c_jail_days'] = (data['c_jail_out'] - data['c_jail_in']).dt.days 

        # if the difference is only hours and the datetimes has same %Y%m%d then -1 retuned 
        # this line below will remove -1 and replace them with 0
        data['c_jail_days'] = data['c_jail_days'].apply(lambda x: 0 if x==-1 else x)

        logging.info('|-> Subtracting datetimes for each entry finished')
        logging.info('|-> Deleting c_jail_in and c_jail_out')
        del data['c_jail_in']
        del data['c_jail_out']
        return data  
    
    def scale(self, data):
        """ 
            For scaling values in the input DataFrame given as argument data. 
            Notice that here we use the PowerTransformer for scaling data. 
        """
        logging.info('|-> Scaling data using Standrand normal scaler ')
        min_max_scaler = preprocessing.PowerTransformer()
        for col in data.columns:
            if col not in ['sex', 'charge']:
                if os.path.exists(f"Scalers/{col}_scaler.pkl"):
                    standard_normalDis_scaler = load(open(f'Scalers/{col}_scaler.pkl', 'rb'))
                    print(f"Want to scale {col} with value data{col}")
                    data[col] = standard_normalDis_scaler.transform(data[col].values.reshape(-1,1))
                else:
                    standard_normalDis_scaler = StandardScaler().fit(data[col].values.reshape(-1,1))
                    data[col] = standard_normalDis_scaler.transform(data[col].values.reshape(-1,1)) 
                    # Save the scaler for the column 
                    dump(standard_normalDis_scaler, open(f'Scalers/{col}_scaler.pkl', 'wb'))        
        logging.info('|-> Scaling data using Standrand normal scaler finished')
        # With this line of code we can use scaler to revert scaled values back to its original values 
        # standard_normalDis_scaler.inverse_transform(data['c_jail_days'])

        return data 

    def flatten(self, data):
        """  
           This method does nothing new - uses only other methods for creating the result flattened table. 

           1. creates categories and calculates custody 
            
        """
        logging.info('|-> Starting to flatten')

        ############## Preprocess #######################
        # Categerize columns given as list 
        data, categories = self.categorize(data, ['sex', 'race', 'charge_degree'])
           
        # pprint(categories )

        # pprint(categories )

        
        # calculate custody period
        data = self.calculate_custody(data)
        # data normalization   
        # self.normalize(data)
        logging.info('|-> Done flattening')
        return data 
    
    def dim_reduction(self, data):

        """ 
            This method uses the dimension reduction algorithm PCA (Principle Component Analysis).
            PCA is helpful in reducing the column sizes, while also keeping almost all possible information from the input columns. 

        """
        logging.info("|-> Dimension reduction using PCA ")
        Y = data['c_jail_days']
        X = data.drop(columns=['c_jail_days'])
        logging.info(f"|-> Deleted column c_jails_days : new dimension is {X.shape}")
        
        pca = PCA(n_components=6)
        if os.path.isfile("models_PCA/pca.pkl"):
            pca = load(open("models_PCA/pca.pkl", "rb"))
            principal_components = pca.transform(X)
        else:
            principal_components = pca.fit_transform(X)
            dump(pca, open('models_PCA/pca.pkl', 'wb'))
        
        principalDF = pd.DataFrame(data=principal_components,
                columns=['PCA_Feature1','PCA_Feature2', 'PCA_Feature3', 'PCA_Feature4', 
                    'PCA_Feature5', 
                    'PCA_Feature6'])
        logging.info(f"|-> Dimension reduced to {principalDF.shape}")
        principalDF['c_jail_days'] = Y 
        return principalDF

    def plot(self, data):
        """ 
            Plots :) 
            You can specify the x and y and plot data as scatter plot. 
            Notice that its helpful to first convert the input data to float values using the methods above and then 
            plot the data to see how values between two columns are distributed. 
            This will also work on PCA data created by dim_reduction method.
        """
        data.plot.scatter(x='sex',y='age', color='green')
        # data.plot.scatter(x='sex_cat', y='priors_count', color='blue')
        # data.plot.scatter(x='priors_count', y='age') 
        # data.plot.scatter(x='priors_count', y='juv_fel_count') 
        # data.plot.scatter(x='id', y='race')
        # data.plot.scatter(x='prison_days', y='sex')
        plt.show()

    def find_correlation(self, data):
        """ 
            This method creates the correlation table for the input DataFrame given as argument data. 
            The correlation_matrix can be printed or plotted as a heatmap. 
        
        """
        logging.info("|-> Finding Correlation of Data")
        
        correlation_matrix = data.corr(method='pearson')

        logging.info("|-> Process of finding correlation is done ")
        logging.info("|-> Plotting correlation matrix")

        sns.heatmap(correlation_matrix, xticklabels= correlation_matrix.columns.values, yticklabels=correlation_matrix.columns.values)
        plt.title("Correlation Matrix")
        plt.show()

    def write_possible_charges(self, data):
        """ 
            Writes DataFrame data given as input into a file called possible_charges.
        """
        di = {}
        group_charge = data['charge'].unique()
        for charge in group_charge:
            di[charge] =  db.calculate_value(db.embed_sentence(charge))

        with open('possible_charges.csv', 'w') as f:
            for k,v in di.items():
                k = k.replace(',', ' ') if isinstance(k, str) else k
                k = k.replace('\n', ' ') if isinstance(k, str) else k
                f.write(f'{k} : {v}, \n')
    
    def describe(self, data):
        logging.info("|-> Data Describtion")
        print(data.describe(percentiles=[.25,.5, .75, .80, .90, .95]).transpose())

    def clear(self, data):
        print("Is Nan : ", np.isnan(data))
        print("Where is Nan : ", np.where(np.isnan(data)))
        for col in data.columns:
            data[col] = data[col].fillna(0)
        print("Is Nan : ", np.isnan(data))
        print("Where is Nan : ", np.where(np.isnan(data)))
        return data


    def first_n_charges(self, data, gr, field_name, added_name):
        #creates n new columns for first n ',' seperatet strings in field_name
        #data, gr: n, field_name:field to split, added_name: name of added columns (added_nameN)
        maxamount = 0
        sum = 0
        cnt = 0
        cntGr = 0
        count_name = added_name + '_count'

        # adding columns for first gr charges
        lastIndex = 0
        for i in range(0, gr):
            loc = data.columns.get_loc(field_name) + i + 1
            name = added_name + str(i)
            data.insert(loc, name, 'x', True)
            lastIndex = loc
        data.insert(lastIndex + 1, count_name, 0, True)
        # end adding

        for ind, row in data.iterrows():
            word = row[field_name]
            if word == word:  # nulltest
                amount = word.count(',') + 1

                singles = word.split(',')
                cap = min(amount, 5)
                for i in range(0, cap):
                    name = added_name + str(i)
                    data.at[ind, name] = singles[i]

                data.at[ind, count_name] = amount

            else:
                amount = 0

            sum += amount
            cnt += 1
            if amount > gr:
                cntGr += 1

            if amount > maxamount:
                maxamount = amount
        sum = sum / cnt
        logging.info(" finished Splitting " + field_name + ":  avg: %0.1f, max: %d, Count of greater than %d : %d of %d" % (sum, maxamount, gr, cntGr, cnt))
        return data

    def processData (self,db, data):
        #expects pandas dataframe in formated like our SQL output

        #split charge lists
        db.first_n_charges(data, 5, 'c_charges', 'charge')
        db.first_n_charges(data, 5, 'r_charges', 'r_charge')
        data = data.drop(columns=['c_charges', 'r_charges'])
        #end split charge lists

        # encode nominal data to numeric values
        tmp = db.categorize(data,
                            ['sex', 'race', 'marital_status', 'charge0', 'charge1', 'charge2', 'charge3', 'charge4',
                             'r_charge0', 'r_charge1', 'r_charge2', 'r_charge3', 'r_charge4', 'score_v_txt',
                             'score_a_txt', 'score_r_txt'])
        data = tmp[0]
        # end encode nominla data

        # encode chargedegrees to vector of counts
        # Einfügen einer Spalte mit der Anzahl an Charge Degrees eines bestimmten Typs
        data['c_(0)'] = data.c_charge_degrees.str.count('(0)')
        data['c_C03'] = data.c_charge_degrees.str.count('(CO3)')
        data['c_CT'] = data.c_charge_degrees.str.count('(CT)')
        data['c_F1'] = data.c_charge_degrees.str.count('(F1)')
        data['c_F2'] = data.c_charge_degrees.str.count('(F2)')
        data['c_F3'] = data.c_charge_degrees.str.count('(F3)')
        data['c_F5'] = data.c_charge_degrees.str.count('(F5)')
        data['c_F6'] = data.c_charge_degrees.str.count('(F6)')
        data['c_F7'] = data.c_charge_degrees.str.count('(F7)')
        data['c_M1'] = data.c_charge_degrees.str.count('(M1)')
        data['c_M2'] = data.c_charge_degrees.str.count('(M2)')
        data['c_M3'] = data.c_charge_degrees.str.count('(M3)')
        data['c_M03'] = data.c_charge_degrees.str.count('(MO3)')
        data['c_NI0'] = data.c_charge_degrees.str.count('(NI0)')
        data['c_TC4'] = data.c_charge_degrees.str.count('(TC4)')
        data['c_TCX'] = data.c_charge_degrees.str.count('(TCX)')
        data['c_X'] = data.c_charge_degrees.str.count('(X)')
        data['c_XXXXXXXXXX'] = data.c_charge_degrees.str.count('XXXXXXXXXX')
        del data['c_charge_degrees']  # Entfernen der entsprechenden Spalte

        data['r_(0)'] = data.r_charge_degrees.str.count('(0)')
        data['r_C03'] = data.r_charge_degrees.str.count('(CO3)')
        data['r_CT'] = data.r_charge_degrees.str.count('(CT)')
        data['r_F1'] = data.r_charge_degrees.str.count('(F1)')
        data['r_F2'] = data.r_charge_degrees.str.count('(F2)')
        data['r_F3'] = data.r_charge_degrees.str.count('(F3)')
        data['r_F5'] = data.r_charge_degrees.str.count('(F5)')
        data['r_F6'] = data.r_charge_degrees.str.count('(F6)')
        data['r_F7'] = data.r_charge_degrees.str.count('(F7)')
        data['r_M1'] = data.r_charge_degrees.str.count('(M1)')
        data['r_M2'] = data.r_charge_degrees.str.count('(M2)')
        data['r_M3'] = data.r_charge_degrees.str.count('(M3)')
        data['r_M03'] = data.r_charge_degrees.str.count('(MO3)')
        data['r_NI0'] = data.r_charge_degrees.str.count('(NI0)')
        data['r_TC4'] = data.r_charge_degrees.str.count('(TC4)')
        data['r_TCX'] = data.r_charge_degrees.str.count('(TCX)')
        data['r_X'] = data.r_charge_degrees.str.count('(X)')
        data['r_XXXXXXXXXX'] = data.r_charge_degrees.str.count('XXXXXXXXXX')
        del data['r_charge_degrees']  # Entfernen der entsprechenden Spalte
        # end encode chargedegrees

        #fill invalid values
        data = data.fillna(0)
        #end fill

        return data

    def processCustom(self, db, data,customData):
        #WARNIGN: sketchy work around, to ensure same Categorization
        #data has to be the unprocessed dataframe
        # custom data should be the new data to processes in the same format
        #!!! CUSTOM DATA MUSN'T CONTAIN NOMINAL VALUES WICH AREN'T ALREADY PRESENT IN DATA
        ndat = data.copy()
        customRowCount = customData.shape[0]
        ndat.append(customData)
        ndat=db.processData(db, ndat)
        pCData= ndat.tail(customRowCount)

        return pCData


if __name__ == '__main__':
    try :
        path = sys.argv[1]
    except:
        logging.error("|-> Please enter the path to csv file")
        sys.exit(0)
    try:

        db   = Database(path)
        data = db.read_csv(path)
     
        # db.describe(data) # prints statistics about each column in data 
        
        # db.describe(data) # prints statistics about each column in data 
        
        ########## standard workflow for new exported data from prisoner database ######################
        #
        #
        #
        #
        data = db.embed_sentences(data, 'charge') # transforms the charge into a meaningfull values
        data = db.flatten(data) # categorizes and calculates custody for input data (only use on data where custody has been not calculated yet)
        data = data.drop(columns=['id']) # drop the id columns because it is not needed anymore 
        data = db.clear(data)
        data = db.dim_reduction(data)
        data = db.scale(data) 
        db.export_csv(data, "reduced1.csv")    
        #
        #
        #
        #
        ################################################################################################
         
        # db.write_possible_charges(data)
        # db.plot(data) 
        # db.describe(data) # prints statistics about each column in data 
        # db.find_correlation(data) # finds correlcation and plots the correlation matrix 
         
    except Exception as e:
        logging.warning(f"|-> Error while loading database {path}: Exception: {e}")
