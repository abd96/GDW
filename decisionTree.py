import Database as DB


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.metrics


#load data
db = DB.Database('rawData.csv')
data= db.read_csv('rawData.csv')
#end load data

#process data
data= db.processData(db,data)
#end process data

#extract labels
v_label = data[['raw_v']].copy()
r_label = data[["raw_r"]].copy()
a_label = data[["raw_a"]].copy()

v_cat = data[['score_v_txt']].copy()
a_cat = data[['score_a_txt']].copy()
r_cat = data[['score_r_txt']].copy()

data = data.drop(columns=['id', 'raw_v', 'dec_v', 'raw_r',
                          'dec_r', 'raw_a', 'dec_a', 'score_v_txt', 'score_r_txt', 'score_a_txt'])
#end extract labels

#db.export_csv(data, "processedData.csv")

def doTree(tdata, tlabel, tcat):
    validator = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=123)
    myTree = tree.DecisionTreeRegressor()
    catTree = tree.DecisionTreeClassifier()

    scores = cross_val_score(myTree, tdata, tlabel, scoring='neg_mean_squared_error', cv=validator)

    print("%0.2f neg mean squared error with a standard deviation of %0.2f (REGRESSION)"
          % (scores.mean(), scores.std()))
    print('label mean value:%0.2f' % (tlabel.mean()))

    Catscores = cross_val_score(catTree, tdata, tcat, scoring='accuracy', cv=validator)
    print("%0.2f accuracy with a standard deviation of %0.2f (CLASSIFIKATION)" % (Catscores.mean(), Catscores.std()))

print("\n\n V prediction:")
doTree(data,v_label,v_cat)
print("label mean: %0.2f" %(v_cat.mean()))
print(v_cat['score_v_txt'].value_counts())
print("\n\n A prediction:")
doTree(data, a_label, a_cat)
print("label mean: %0.2f" %(a_cat.mean()))
print(a_cat['score_a_txt'].value_counts())
print("\n\n R prediction:")
doTree(data, r_label, r_cat)
print("label mean: %0.2f" %(r_cat.mean()))
print(r_cat['score_r_txt'].value_counts())


'''
#generating visualisation of tree for analysing purposes
nameList = (
'sex', 'race', 'age', 'marital_status', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
'is_recid', 'is_violent_recid', 'c_custody_in_days', 'charge0', 'charge1', 'charge2', 'charge3', 'charge4',
'charge_count', 'r_custody_in_days', 'r_charge0', 'r_charge1', 'r_charge2', 'r_charge3', 'r_charge4', 'r_charge_count',
'h_prison', 'h_jail', 'rec_supervision_level', 'c_(0)', 'c_C03', 'c_CT', 'c_F1', 'c_F2', 'c_F3', 'c_F5', 'c_F6', 'c_F7',
'c_M1', 'c_M2', 'c_M3', 'c_M03', 'c_NI0', 'c_TC4', 'c_TCX', 'c_X', 'c_XXXXXXXXXX', 'r_(0)', 'r_C03', 'r_CT', 'r_F1',
'r_F2', 'r_F3', 'r_F5', 'r_F6', 'r_F7', 'r_M1', 'r_M2', 'r_M3', 'r_M03', 'r_NI0', 'r_TC4', 'r_TCX', 'r_X',
'r_XXXXXXXXXX')

catTree = tree.DecisionTreeClassifier()
catTree.fit(data,v_cat)
tree.export_graphviz(catTree,out_file='treegraph_2.dot',leaves_parallel=True,feature_names=nameList)
#end visualisation
'''


print("\nend of Operation")