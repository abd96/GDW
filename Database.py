import sys
import csv 
import sqlite3 
import logging 
import traceback 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing 


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
        data.to_csv(name)

    def categorize(self, data, cols):
        for col in cols:
            data[col] = data[col].astype('category')
            data[col+"_cat"] = data[col].cat.codes
            del data[col] 
        self.export_csv(data, 'data_cat.csv')
        return data 
    
    def calculate_custody_persion(self, data):
        data['in_custody']     = pd.to_datetime(data['in_custody'])
        data['out_custody']    = pd.to_datetime(data['out_custody'])
        data['custody_period'] = data['out_custody'].sub(data['in_custody']).dt.days
        del data['in_custody']
        del data['out_custody'] 
        self.export_csv(data, 'data_cat.csv')
        
    def normalize(self, data):
        x = data.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

    def flatten(self, data):
        logging.info('|-> Starting to flatten')

        ############## Preprocess #######################
        # Categerize columns given as list 
        data = self.categorize(data, ['sex', 'race', 'charge', 'charge_degree'])
        
        # calculate custody period
        self.calculate_custody_persion(data)
        # data normalization   
        # self.normalize(data)
        logging.info('|-> Done flattening')
        
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
        data.plot.scatter(x='sex', y='priors_count', color='blue')
        data.plot.scatter(x='priors_count', y='age') 
        data.plot.scatter(x='priors_count', y='juv_fel_count') 
        plt.show()

if __name__ == '__main__':
    try :
        path = sys.argv[1]
    except:
        logging.error("|-> Please enter the path to csv file")
        sys.exit(0)
    try:

        db = Database(path)
        data = db.read_csv(path)
        # db.plot(data) 
        # db.export_csv(data, "data.csv")
        db.flatten(data)
         
    except Exception as e:
        logging.warning(f"|-> Error while loading database {path}: Exception: {e}")
        traceback.print_exc()

