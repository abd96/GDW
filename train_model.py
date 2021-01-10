import sys 
import csv 
import logging 
import pandas as pd 
import numpy as np 
from pickle import load 
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


### Local 
from Database import Database 


### Logging 
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
    print(model.summary())
    return model  


def plot_history(history):
    fig, axs = plt.subplots(2)
    fig.suptitle("Training History")

    axs[0].plot(history.history['mse'])
    axs[0].plot(history.history['mae'])
    axs[1].plot(history.history['mape'])
    axs[0].plot(history.history['cosine_proximity'])
    axs[0].legend(['mse', 'mae','cosine_proximity'], loc='upper right')
    axs[1].legend(['mape'], loc='upper right')
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Error')
    plt.savefig('train_history.png')
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
    json_file = open('models_ANN/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models_ANN/model.h5")
    return loaded_model

def build_model(X_train, X_test, Y_train, Y_test):
    model = create_model(X_train, Y_train)
    history = model.fit(X_train, Y_train, epochs=155)

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

def preprocess(path):
    db = Database(path)
    
    data = db.read_csv(path)
    data = db.dim_reduction(data)
    data = db.scale(data)
    return data

def inverse_scale_prediction(prediction):
    # load Scaler 
    scaler = load(open("Scalers/c_jail_days_scaler.pkl", 'rb'))
    return scaler.inverse_transform(prediction)


def train(path="reduced1.csv"):
    #################### Training ##########################################
    X_train, Y_train, X_test, Y_test = split(path)
    build_model(X_train, X_test, Y_train, Y_test) 
    evaluate(path, X_test, Y_test) 

def split(path):
    data = read_csv(path)
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
    
    return X_train, Y_train, X_test, Y_test  

def evaluate(path, X_test, Y_test):
    # Evaluate on test Data 
    model = load_model()
    model.compile(loss='mean_absolute_error', optimizer='adam',
            metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
    score = model.evaluate(X_test, Y_test)
    print();print('Score on Test Data ')
    for i in range(len(model.metrics_names)):
        print(f'{model.metrics_names[i]}: {score[i] }')
    
def main():
    mode = ""
    path = ""
    try:
        mode  = sys.argv[1]
        path  = sys.argv[2]
    except: 
        if path == "" or mode == "":
            logging.error("|-> mode not scpecified. Please enter *train*, *test* or *evaluate* as first argument")
            logging.error("|-> Please enter mode and the path to csv file")
            
    if mode == 'test':
        input_data = preprocess(path).drop(columns=['c_jail_days'])
        model = load_model()
         
        prediction = inverse_scale_prediction(model.predict(input_data))[0][0]
        print();print("******************************************************************")
        print("Period prediction for input : ", int(prediction), 'days') 
        print("******************************************************************")
        sys.exit(0) 
    elif mode == "train":
        train(path)

    elif mode == "evaluate":
        _, _, X_test, Y_test  = split(path)
        evaluate(path, X_test, Y_test)

if __name__ == '__main__':
    main()

