# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:12:30 2020

@author: kedar.kulkarni

Program: Kaggle Competetion - Titanic
"""

#Importing important libraries
import numpy as np
import pandas as pd

#Read the input CSV file
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data['Embarked'] = train_data['Embarked'].fillna('N')
test_data['Fare'] = test_data['Fare'].fillna(0)

#Splitting the dataset into dependent and independent dataset
X_train = train_data.iloc[:, [2, 4, 6, 7, 9, 11]].values
y_train = train_data.iloc[:, 1].values
X_test = test_data.iloc[:, [1, 3, 5, 6, 8, 10]].values

#Convert the lables to appropriate values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()
X_train[:, 1] = labelEncoder_X.fit_transform(X_train[:, 1])
X_train[:, 5] = labelEncoder_X.fit_transform(X_train[:, 5])

labelEncoder_X_test = LabelEncoder()
X_test[:, 1] = labelEncoder_X_test.fit_transform(X_test[:, 1])
X_test[:, 5] = labelEncoder_X_test.fit_transform(X_test[:, 5])

'''
ohe_X = OneHotEncoder(categorical_features=[5])
X_train = ohe_X.fit_transform(X_train).toarray()

ohe_X_test = OneHotEncoder(categorical_features=[5])
X_test = ohe_X_test.fit_transform(X_test).toarray()


transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [5])],remainder='passthrough')
X_train = np.array(transformer.fit_transform(X_train), dtype=np.float)
X_test = np.array(transformer.fit_transform(X_test), dtype=np.float)

#Avoiding the Dummy Variable Trap. Removing 1st column since it is independent
X_train = X_train[:, 1:]
#X_test = X_test[:, 1:]

#Perform Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
'''

#Fitting the RANDOM FOREST model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

'''
#Fitting the SVM Classification
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


# Fitting XGBoost to the Training set
from xgboost. import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
'''

#Predicting the results
y_pred = classifier.predict(X_test)

print(classifier.score(X_train, y_train))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

