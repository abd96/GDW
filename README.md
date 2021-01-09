# Penalty prediction for prisoners based on person's criminal history
In this project the database from [compas-analysis](https://github.com/propublica/compas-analysis)
has been used in order to predict the prisoners penalty based on multiple features that has been 
choosen wisely.  

After further discussions, the following features standed out from all other 
features and were considered for training :  

| sex  |  race | age  | juv_fel_count  |  priors_count | jubv_misd_count |in_custody|
|---|---|---|---|---|---|---|

| out_custody|  juv_other_count| c_jail_in | c_jail_out  |  charge  | charge_degree |
|---|---|---|---|---|---|---|

| prison_in|  prison_out |
|---|---|---|---|---|---|


In order to be able to train a prediction model using these data, mutliple preprocessing steps were necessary. 
Firstly we take a look at the preprecossing steps done in script `Database.py`. 
Secondly we check out the different models that we used. We finally then evaluate each model on same 
data and compare the results.

### Step 1: Preprocessing 

##### Charge conversion
The most important and hardest step was to preprocess the column *charge*. In this column, data were
give as strings. Example for charges are `Possess Cannabis/20 Grams Or Less` and 
`Possession of Cannabis`. As you may see from these two charges, these are similar charges, thus
our idea was to preprocess these sentences such that for similiar charges we get a similiar 
representation of these charges.

For the first step we converted the sentences into tokens. After that we removed all stopwords and special characters,
that are not important for the meaning of the sentences. Then we transformed each token 
in the tokens list into a vector embedding using [Gensim](https://pypi.org/project/gensim/). After that 
we added each vector for each sentence the set of word embeddings and took the arithmetic mean of the remaining 
values. Notice that gensim creates a fixed length vectors. the model can be found in `models_gensim`.

For the specified examples above, the following values were achieved : 

* *Possess Cannabis/20 Grams Or Less* : 0.0456
* *Possession of Cannabis* : 0.0781

Maybe testing with Jaccard, cosine similarity is also not a bad idea.


##### Categorizing Sex, race and charge_degree 

Nothing too special here we only turned theses columns into categories, you can find 
how the categories looks like in folder `possible_categories`

##### Calculating Periods for custody_in, custody_out, c_jail_in, c_jail_out, prison_in, prison_out 
Here we only calculated the period using the in's and the out's in days  

Here is an example of the data at this point:


| sex  |  race | age  | juv_fel_count  |  priors_count | juv_misd_count | juv_other_count |
|---|---|---|---|---|---|---|
| 1|  0 | 60  | 0  | 11 | 0 | 0 |

| charge | charge degree| jail_days  | prison_days  | c_jail_days |
|---|---|---|---|---|---|---|
| -0.0493| 9 | 4  | 343  | 33 |

Basically our label is c_jail_days

##### Reducing the demension  
For dimension reduction we used Principle Components Analysis (PCA) to reduce the dimension of input data. The best 
number of componenets was 6. 
You can find the PCA model in models_PCA

##### Scaling data using the 
For each training feature the Scaler [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) was using and dumped in Scalers folder. 

### Step 2: Training 

##### Artificial Neural Network (ANN)
TODO

### Step 3: Evaluating 
TODO 

### Testing on single input 
In `input.csv` you can fill the input values for testing. 
Running `python3 train_model test input.csv` will yield the prediction of the period from 
the trained model in `models_ANN`

### Scripts excution and installing requirements
TODO


