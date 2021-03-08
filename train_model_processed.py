import pandas as pd 
import numpy as np 
from pickle import load 
from keras.models import model_from_json
from tensorflow import keras 
from tensorflow.keras import layers 
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier as KC
from keras.wrappers.scikit_learn import KerasRegressor as KR
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression 
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import kerastuner as kt 
from kerastuner import HyperModel

### Local 
from Database import Database 

class RegressionHyperModel(HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        units = [1024, 512, 128, 64, 32]
        model.add(
                layers.Dense(
                        units = 2048,
                        input_shape = self.input_shape,
                        activation = hp.Choice('input_layer', values=['relu', 'linear'])
                        )
                )
        for unit in units:
            model.add(
                    layers.Dense(
                            units = unit, 
                            kernel_initializer = 'normal',
                            activation = hp.Choice('dense__activation', values=['relu', 'linear', 'sigmoid'], default='relu')
                            )
                    )
        
        model.add(
                layers.Dense(
                        units = 3,
                        activation = hp.Choice('output_layer', values=['linear', 'sigmoid', 'relu'])
                        )
                )
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 0.5, 0.1])), loss='mean_absolute_error', metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
        return model

def create_model(X_train, Y_train):
    model = keras.Sequential([
        Dense(2048, input_dim=X_train.shape[1], activation='relu'),
        Dense(1024, kernel_initializer='normal', activation='relu'),
        Dense(512, kernel_initializer='normal', activation='relu'),
        Dense(128, kernel_initializer='normal', activation='relu'),
        Dense(64, kernel_initializer='normal', activation='relu'),
        Dense(32, kernel_initializer='normal', activation='relu'),
        Dense(Y_train.shape[1], activation='linear')
        ])
    model.compile(loss='mean_absolute_error', optimizer='adam',
            metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
    print(model.summary())
    return model  


def build_model(X_train, Y_train, X_test, Y_test):
    # model = create_model(X_train, Y_train)
    hyperModel = RegressionHyperModel((X_train.shape[1],))
     

    tuner_rs = RandomSearch(hyperModel, objective='mse', seed=42, max_trials=135, 
            executions_per_trial=1 ,directory='param_opt_checkouts', project_name='GDW')

    tuner_rs.search(X_train, Y_train, epochs=155)
    
    print("Best Results")
    best_model = tuner_rs.get_best_models(num_models=1)[0]
    print("!!!!!!!!!!!!!!!!!!!!!Evaluation ON TEST DATA!!!!!!!!!!!!!!!!!!!!!!") 
    metrics = ['loss', 'mse', 'mae', 'mape', 'cosine_proximity'] 
    print(best_model.summary())
    _eval = best_model.evaluate(X_test, Y_test)
    print(_eval)
    for i in range(len(metrics)):
        print(f'{metrics[i]} : {_eval[i]}')
    tuner_rs.results_summary()
    best_model.save('./models_ANN/best_model')
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 

    #history = model.fit(X_train, Y_train, epochs=155)

    # evaluate model 
    # scores = model.evaluate(X_train, Y_train)
            
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # plot_history(history)
    #save_model(model)



# mode can be regression or calssification
def split(path, mode='regression'):

    db = Database(path)
    
    data = db.read_csv(path)

    if mode ==  "regression" : 
        print("Splitting data into test and training ")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
        print("Train data of shape : ", train_data.shape)
        print("Test data of shape : ", test_data.shape)

        Y_train = train_data[['raw_v', 'raw_r', 'raw_a']]
        X_train = train_data.drop(columns= ['raw_v', 'raw_r', 'raw_a']) 
        print("X_train has shape of :", X_train.shape)
        print("Y_train has shape of :", Y_train.shape)
        
        Y_test = test_data[['raw_v', 'raw_r', 'raw_a']]
        X_test = test_data.drop(columns= ['raw_v', 'raw_r', 'raw_a']) 
        print("X_test has shape of :", X_test.shape)
        print("Y_test has shape of :", Y_test.shape)
        
        build_model(X_train, Y_train, X_test, Y_test)

    elif mode == "classification":
        y = data[['score_v_txt', 'score_r_txt', 'score_a_txt']]
        data = data.drop(columns= ['score_v_txt', 'score_r_txt', 'score_a_txt']) 
        pass         

def main():
    path = './ProcesssedData.csv'
    
    split(path) 
    

if __name__ == '__main__':
    main()
