import sys
import csv 
import re
import sqlite3 
import logging 
import traceback 
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from pprint import pprint 
from sklearn import preprocessing 
from datetime import datetime 
from gensim.models import word2vec


# NLP 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer

logging.getLogger().setLevel(logging.INFO)

class Database: 

    def __init__(self, fileDB):
        self.fileDB = fileDB 
        self.conn = self.create_connection() 
        self.model = word2vec.Word2Vec.load("word2vec.model")


    def create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.fileDB)
        except Exception as e:
            logging.warning(f"Error creating Connection to Database {self.fileDB}: Exception: {e}")

        return conn 

    def list_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [x[0] for x in cursor.fetchall() if x[0] != 'summary']
    
    def get_columns(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        return cursor.fetchall()

    def get_columns_names(self, table_name):
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
        logging.info(f"|-> Reading path {path}")
        data = pd.read_csv(path) 
        logging.info(f"|-> Successfully read data of shape {data.shape}")        
        return data
    
    def export_csv(self, data, name):
        logging.info(f"|-> Exporting Data to csv, filename : {name}")
        data.to_csv(name)
        logging.info(f"|-> Done writing data to csv, filename : {name}")
    
    def embed_sentence(self, sent):
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

    def normalize(self, data):
        x = data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

    def flatten(self, data):
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

    def join_to_csv(self):
        cursor = self.conn.cursor()
        # Die Ergebnisse dieser Query sind auch durch sqlitebrowser als csv speicherbar, aber die Methode könnte später auch hilfreich sind.
        query = " \
                SELECT DISTINCT *  \
                FROM ( \
                SELECT DISTINCT A.*, charge, charge_degree  \
                FROM ( \
                        SELECT P.id,P.sex,P.race,P.age,P.juv_fel_count, P.priors_count, P.juv_misd_count, \
                                JH.in_custody, JH.out_custody, P.juv_other_count \
                        FROM people as P \
                        JOIN jailhistory as JH   \
                        ON P.id = JH.person_id \
                    ) AS A \
                JOIN charge as C \
                On A.id = C.person_id ) AS B\
                "
        cursor.execute(query)

        self.export_csv(cursor.fetchall(), 'data.csv')
    
    def plot(self, data):
        data.plot.scatter(x='sex',y='age', color='green')
        # data.plot.scatter(x='sex_cat', y='priors_count', color='blue')
        # data.plot.scatter(x='priors_count', y='age') 
        # data.plot.scatter(x='priors_count', y='juv_fel_count') 
        # data.plot.scatter(x='id', y='race')
        # data.plot.scatter(x='prison_days', y='sex')
        plt.show()


    def describe(self, data, column):
        logging.info(f'|-> Describtion of {column}')
        print(data[column].describe())
        
    def find_correlation(self, data):
        logging.info("|-> Finding Correlation of Data")
        print(data.corr(method='pearson'))
        logging.info("-> Process of finding correlation is done ")

    def write_possible_charges(self, data):
        di = {}
        group_charge = data['charge'].unique()
        for charge in group_charge:
            di[charge] =  db.calculate_value(db.embed_sentence(charge))

        with open('possible_charges.csv', 'w') as f:
            for k,v in di.items():
                k = k.replace(',', ' ') if isinstance(k, str) else k
                k = k.replace('\n', ' ') if isinstance(k, str) else k
                f.write(f'{k} : {v}, \n')

if __name__ == '__main__':
    try :
        path = sys.argv[1]
    except:
        logging.error("|-> Please enter the path to csv file")
        sys.exit(0)
    try:

        db = Database(path)
        data = db.read_csv(path)
        # data = db.embed_sentences(data, 'charge')
        # data = db.flatten(data)
        # db.export_csv(data, "dataaaaaaaaaaaa.csv")    
        # db.write_possible_charges(data)
        # data = db.embed_sentence(data, 'charge')
        # data = db.flatten(data)
        # db.export_csv(data, "data_cat_new_new.csv")    
        # db.plot(data) 
        # db.export_csv(data, "data.csv")
        # db.flatten(data)
        # db.plot(data)
        # db.find_correlation(data)
        # db.describe(data, 'race')
         
    except Exception as e:
        logging.warning(f"|-> Error while loading database {path}: Exception: {e}")
        traceback.print_exc()

