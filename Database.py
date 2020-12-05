import sys
import csv 
import sqlite3 
import logging 
import pandas as pd
import matplotlib.pyplot as plt 


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
    
    def export_csv(self, data):

        cursor = self.conn.cursor()
        csv_writer = csv.writer(open('falttened_data.csv', 'w'))
        csv_writer.writerow(("id","sex",'race','age','juv_fel_count','priors_count','juv_misd_count', 'out_custody', 'in_custody','juv_other_count', 'charge', 'charge_degree'))
        for x in data:
            csv_writer.writerow(x)

    def flatten(self, data):
        logging.info('|-> Starting to flatten')

        # get all column names  
        col_names = [col for col in data.columns]
    
    
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

        self.export_csv(cursor.fetchall())
    
    def plot(self, data):
        data.plot.scatter(x='sex',y='age', color='green')
        data.plot.scatter(x='sex', y='priors_count', color='blue')
        data.plot.scatter(x='priors_count', y='age') 
        data.plot.scatter(x='priors_count', y='juv_fel_count') 
        plt.show()

if __name__ == '__main__':
    path = '/mnt/c/Users/Abdul/Desktop/prisoner.db'
    try :
        path = sys.argv[1]
    except:
        logging.error("|-> Please enter the path to csv file")
        sys.exit(0)
    try:

        db = Database(path)
        data = db.read_csv('data.csv')
        db.plot(data) 

    except Exception as e:
        logging.warning(f"|-> Error while loading database {path}: Exception: {e}")

