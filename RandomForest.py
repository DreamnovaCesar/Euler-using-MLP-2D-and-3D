import matplotlib.pyplot as plt
from sklearn import datasets
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from S_3D_Article_General import Input_3D_array
from S_3D_Article_General import Output_3D_array

#
# Create an instance of Random Forest Classifier
#
forest = RandomForestClassifier(criterion = 'gini',
                                 n_estimators = 10,
                                 random_state = 2,
                                 n_jobs = 10)
#
# Fit the model
#
forest.fit(Input_3D_array, Output_3D_array)
 
#
# Measure model performance
#

y_pred = forest.predict(Input_3D_array)

print(Output_3D_array)
print('\n')
print(y_pred)

print((y_pred == Output_3D_array).all())

joblib.dump(forest, "forest.joblib")

Tres = forest.predict([[0, 0, 0, 0, 0, 0, 1, 0]])

print(Tres)
print(Tres)

