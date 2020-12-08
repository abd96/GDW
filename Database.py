import sys
import csv 
import sqlite3 
from tqdm import tqdm 
import logging 
import traceback 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing 

from gensim.models import word2vec

logging.getLogger().setLevel(logging.INFO)

class Database: 

    def __init__(self, fileDB):
        self.fileDB = fileDB 
        self.conn = self.create_connection() 

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

    def embed_sentence(self, data, col):
        ''' 
            step 1) train a word2vec model on all possible words in charge column 
            step 2) convert all possible words in each row in data[col] to a vector using that model 
            step 3) muliply all vectors of each word of charge text with each others 
            step 4) normalize the product vector 
            step 5) save the product as a sentence representation of the charge 

        '''
        corpus = data[col].values.astype('U').tolist()
        corpus = [sentence.split() for sentence in corpus]
        model = word2vec.Word2Vec(corpus, min_count=1)
        # model.save("word2vec.model")
        model = word2vec.Word2Vec.load("word2vec.model") # this is how you load word2vec model
        corpus_sent2vec = []
        for tokenized_charge in tqdm(corpus):
            #logging.info(f"|-> Working on word : {tokenized_charge}")
            product = np.ones(shape=(100,)) 
            for word in tokenized_charge:
                vector_of_word = model.wv[word]
                product = np.dot(product,vector_of_word)
                #logging.info("|-> Converted word <{word}> to vector and multiplying ")
            #logging.info("|-> Result of multiplication ")
            # scale 
            corpus_sent2vec.append(product)
                 
        data[col] = corpus_sent2vec
        return data 

    def categorize(self, data, cols):
        logging.info("|-> Starting to categorize ")
        for col in cols:
            logging.info(f"|-> categorizing {col}")
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
        logging.info("|-> Done categorizing ")
        return data 
    
    def calculate_custody(self, data, output_file_name):
        logging.info('|-> Converting jail custody to datetime')
        data['in_custody']     = pd.to_datetime(data['in_custody'])
        data['out_custody']    = pd.to_datetime(data['out_custody'])
        logging.info('|-> Converting jail custody to datetime finished ')
        logging.info('|-> Subtracting datetimes for each entry ')
        data['jail_days'] = data['out_custody'].sub(data['in_custody']).dt.days
        data['jail_hours'] = data['out_custody'].sub(data['in_custody']).dt.components['hours']
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
        data['prison_hours'] = data['prison_out'].sub(data['prison_in']).dt.components['hours']
        logging.info('|-> Subtracting datetimes for each entry finished ')
        logging.info('|-> Deleting in_custody and out_custody')
        del data['prison_in']
        del data['prison_out'] 
        logging.info(f"|-> Saving Data to {output_file_name}")
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
        data = self.categorize(data, ['sex', 'race', 'charge_degree'])
        
        # calculate custody period
        data = self.calculate_custody(data, 'data_cat_new.csv')
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
        data.plot.scatter(x='sex_cat',y='age', color='green')
        data.plot.scatter(x='sex_cat', y='priors_count', color='blue')
        data.plot.scatter(x='priors_count', y='age') 
        data.plot.scatter(x='priors_count', y='juv_fel_count') 
        plt.show()
    
    def find_correlation(self, data):
        logging.info("|-> Finding Correlation of Data")
        print(data.corr(method='pearson'))
        logging.info("-> Process of finding correlation is done ")

if __name__ == '__main__':
    try :
        path = sys.argv[1]
    except:
        logging.error("|-> Please enter the path to csv file")
        sys.exit(0)
    try:

        db = Database(path)
        data = db.read_csv(path)
        data = db.flatten(data)
        data = db.embed_sentence(data, 'charge')
        db.export_csv(data, "data_cat_new_new.csv")    
        # db.plot(data) 
        # db.export_csv(data, "data.csv")
        # db.flatten(data)
        # db.plot(data)
        # db.find_correlation(data)
         
    except Exception as e:
        logging.warning(f"|-> Error while loading database {path}: Exception: {e}")
        traceback.print_exc()

