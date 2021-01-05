import logging 
import pandas as pd 
import numpy as np 

from keras.models import model_from_json
from matplotlib import pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier as KC
from keras.wrappers.scikit_learn import KerasRegressor as KR
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression 
from kerastuner.tuners import RandomSearch

logging.getLogger().setLevel(logging.INFO)

def create_model(X_train, Y_train):
    model = keras.Sequential([
        Dense(1024, input_dim=X_train.shape[1], activation='relu'),
        Dense(512, kernel_initializer='normal', activation='relu'),
        Dense(128, kernel_initializer='normal', activation='relu'),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dense(32, kernel_initializer='normal', activation='relu'),
        Dense(1, activation='linear')
        ])
    model.compile(loss='mean_absolute_error', optimizer='adam',
            metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
    return model  


def plot_history(history):
    plt.plot(history.history['mse'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['mape'])
    plt.plot(history.history['cosine_proximity'])
    plt.show()

def save_model(model):
    logging.info("|-> Saving trained model ")
    # serialize model to json 
    model_json = model.to_json()
    with open("model.json", 'w') as f:
        f.write(model_json)
    # serialize model to HDF5
    model.save_weights("model.h5")
    logging.info("|-> Model saved to model.json and its weights to model.h5")

def load_model():
    #load json and create model 
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    return loaded_model

def build_model(X_train, X_test, Y_train, Y_test):
    model = create_model(X_train, Y_train)
    history = model.fit(X_train, Y_train, epochs=150)

    # evaluate model 
    # scores = model.evaluate(X_train, Y_train)
            
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    plot_history(history)
    save_model(model)

def read_csv(path):
    logging.info(f"|-> Reading path {path}")
    data = pd.read_csv(path) 
    logging.info(f"|-> Successfully read data of shape {data.shape}")        
    return data

def main():
    data = read_csv("reduced1.csv")
    logging.info("|-> Splitting dataset for train and test ")
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset  = data.drop(train_dataset.index)
    logging.info(f"|-> Splitting the data is done | train_set of shape {train_dataset.shape},  test_dataset of shape {test_dataset.shape}") 
    
    logging.info("|-> Splitting features from labels ")
    X_train = train_dataset.copy()
    X_test  = test_dataset.copy()
    
    Y_train = X_train.pop('c_jail_days')
    Y_test  = X_test.pop('c_jail_days')

    logging.info('|-> Splitting features from labels done ')
    logging.info(f'|-> X_train shape : {X_train.shape}')
    logging.info(f'|-> X_test  shape : {X_test.shape}')
    logging.info(f'|-> Y_train shape : {Y_train.shape}')
    logging.info(f'|-> Y_test  shape : {Y_test.shape}')
   
    
    #################### Training ##########################################
    # build_model(X_train, X_test, Y_train, Y_test) 
   
    #################### Testing ##########################################
    # Evaluate on test Data 
    model = load_model()
    model.compile(loss='mean_absolute_error', optimizer='adam',
            metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
    score = model.evaluate(X_test, Y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == '__main__':
    main()

