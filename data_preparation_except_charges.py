import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Laden der Daten. Erstellen von Kopie zur Bearbeitung
dataset_1 = pd.read_csv('sql_export.csv', sep = ';')
dataset = dataset_1.select_dtypes(include=['object', 'float64', 'int64']).copy()

dataset = dataset.fillna(-1) # NULL durch -1 ersetzen

"""
--- OBSOLETE ABFRAGEN ---

# Informationen über die Daten
print(dataset.info()) # 

# Anfügen von c_charge_degrees & r_charge_degrees als eigene Spalte
dataset = pd.concat([dataset, dataset['c_charge_degrees'].str.split(',', expand=True)], axis=1)

print(dataset['sex'].value_counts())
Male      9244
Female    2405

print(dataset['race'].value_counts())
African-American    5747
Caucasian           4056
Hispanic            1090
Other                658
Asian                 58
Native American       40

# print(dataset['marital_status'].value_counts())
Single               8898
Married              1427
Divorced              517
Significant Other     396
Separated             282
Unknown                70
Widowed                59
"""

# Categorical Features auf Zahlen mappen
replace_map = {'sex': {'Male': 1, 'Female': 2},
               'race': {'African-American': 1, 'Caucasian': 2, 'Hispanic': 3, 'Other': 4, 'Asian': 5,
                        'Native American': 6},
               'marital_status': {'Single': 1, 'Married': 2, 'Divorced': 3, 'Significant Other': 4, 'Separated': 5,
                        'Unknown': 6, 'Widowed': 7}}

dataset.replace(replace_map, inplace=True)

# Einfügen einer Spalte mit der Anzahl an Charge Degrees eines bestimmten Typs
dataset['c_(0)'] = dataset.c_charge_degrees.str.count('(0)')
dataset['c_C03'] = dataset.c_charge_degrees.str.count('(CO3)')
dataset['c_CT'] = dataset.c_charge_degrees.str.count('(CT)')
dataset['c_F1'] = dataset.c_charge_degrees.str.count('(F1)')
dataset['c_F2'] = dataset.c_charge_degrees.str.count('(F2)')
dataset['c_F3'] = dataset.c_charge_degrees.str.count('(F3)')
dataset['c_F5'] = dataset.c_charge_degrees.str.count('(F5)')
dataset['c_F6'] = dataset.c_charge_degrees.str.count('(F6)')
dataset['c_F7'] = dataset.c_charge_degrees.str.count('(F7)')
dataset['c_M1'] = dataset.c_charge_degrees.str.count('(M1)')
dataset['c_M2'] = dataset.c_charge_degrees.str.count('(M2)')
dataset['c_M3'] = dataset.c_charge_degrees.str.count('(M3)')
dataset['c_M03'] = dataset.c_charge_degrees.str.count('(MO3)')
dataset['c_NI0'] = dataset.c_charge_degrees.str.count('(NI0)')
dataset['c_TC4'] = dataset.c_charge_degrees.str.count('(TC4)')
dataset['c_TCX'] = dataset.c_charge_degrees.str.count('(TCX)')
dataset['c_X'] = dataset.c_charge_degrees.str.count('(X)')
dataset['c_XXXXXXXXXX'] = dataset.c_charge_degrees.str.count('XXXXXXXXXX')
del dataset['c_charge_degrees'] # Entfernen der entsprechenden Spalte

dataset['r_(0)'] = dataset.r_charge_degrees.str.count('(0)')
dataset['r_C03'] = dataset.r_charge_degrees.str.count('(CO3)')
dataset['r_CT'] = dataset.r_charge_degrees.str.count('(CT)')
dataset['r_F1'] = dataset.r_charge_degrees.str.count('(F1)')
dataset['r_F2'] = dataset.r_charge_degrees.str.count('(F2)')
dataset['r_F3'] = dataset.r_charge_degrees.str.count('(F3)')
dataset['r_F5'] = dataset.r_charge_degrees.str.count('(F5)')
dataset['r_F6'] = dataset.r_charge_degrees.str.count('(F6)')
dataset['r_F7'] = dataset.r_charge_degrees.str.count('(F7)')
dataset['r_M1'] = dataset.r_charge_degrees.str.count('(M1)')
dataset['r_M2'] = dataset.r_charge_degrees.str.count('(M2)')
dataset['r_M3'] = dataset.r_charge_degrees.str.count('(M3)')
dataset['r_M03'] = dataset.r_charge_degrees.str.count('(MO3)')
dataset['r_NI0'] = dataset.r_charge_degrees.str.count('(NI0)')
dataset['r_TC4'] = dataset.r_charge_degrees.str.count('(TC4)')
dataset['r_TCX'] = dataset.r_charge_degrees.str.count('(TCX)')
dataset['r_X'] = dataset.r_charge_degrees.str.count('(X)')
dataset['r_XXXXXXXXXX'] = dataset.r_charge_degrees.str.count('XXXXXXXXXX')
del dataset['r_charge_degrees'] # Entfernen der entsprechenden Spalte

dataset = dataset.fillna(0) # Auffüllen mit Nullen, falls kein Vergehen vorliegt


dataset.to_csv(r'D:\Users\bud\Documents\PyCharm\KNN\prepared_data.csv', sep=';') ## Ausgabe als CSV