from tensorflow import keras 
from tensorflow.keras import layers 
from keras.layers import Dense
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
import kerastuner as kt 
from kerastuner import HyperModel
import matplotlib.pyplot as plt 
from train_model import save_model, load_model, plot_history
### Local 
from Database import Database 

class RegressionHyperModel(HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        units = [1024, 512, 64, 32,16]
        model.add(
                layers.Dense(
                        units = 2048,
                        input_shape = self.input_shape,
                        activation = hp.Choice('input_layer', values=['linear'])
                        )
                )
        for unit in units:
            model.add(
                    layers.Dense(
                            units = unit, 
                            kernel_initializer = 'normal',
                            bias_initializer   = 'zeros',
                            activation = hp.Choice('dense__activation', values=['relu'])
                            )
                    )
        model.add(
                layers.Dense(
                        units = 3,
                        activation = hp.Choice('output_layer', values=['linear'])
                        )
                )
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[0.001])), loss='mean_absolute_error', metrics=['mse', 'mae', 'mape', 'cosine_proximity'])
        return model


def predict(model):
    path = "./rawData.csv";
    db   = Database(path); 
    data = db.read_csv(path)
    customData = db.read_csv("./customData.csv")
    processed_input_data = db.processCustom(db, data, customData)
    print("Processed Data -> ")
    X_input = processed_input_data.drop(columns= ['id', 'raw_v', 'raw_r', 'raw_a'])
    prediction = model.predict(X_input) 
    print("PREDICTION ", prediction)

def build_model(X_train, Y_train, X_test, Y_test):
    hyperModel = RegressionHyperModel((X_train.shape[1],))
     

    tuner_rs = RandomSearch(hyperModel, objective='mse', max_trials=135, 
            executions_per_trial=1 ,directory='param_opt_checkouts', project_name='GDW')
    tuner_rs.search(X_train, Y_train, validation_data=(X_test, Y_test), epochs=160)
    best_model = tuner_rs.get_best_models(num_models=1)[0]
    
    #metrics = ['loss', 'mse', 'mae', 'mape', 'cosine_proximity'] 
    #_eval = best_model.evaluate(X_test, Y_test)
    #print(_eval)
    #for i in range(len(metrics)):
    #    print(f'{metrics[i]} : {_eval[i]}')
    
    # history = best_model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=50)

    # best_model.save('./models_ANN/best_model')
    
    # save_model(best_model)   
    tuner_rs.results_summary()
    print(load_model().summary())
    predict(best_model);

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
        X_train = train_data.drop(columns= ['id', 'Unnamed: 0','raw_v', 'raw_r', 'raw_a']) 
         
        Y_test = test_data[['raw_v', 'raw_r', 'raw_a']]
        X_test = test_data.drop(columns= ['id', 'Unnamed: 0','raw_v', 'raw_r', 'raw_a']) 
        
        print("X_train has shape of :", X_train.shape)
        print("Y_train has shape of :", Y_train.shape)
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
