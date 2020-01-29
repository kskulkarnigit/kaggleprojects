# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:07:28 2020

@author: kedar.kulkarni

Program: Kaggle Competetion - Titanic Survivor Program
"""

#Importing important libraries
import numpy as np
import pandas as pd
import os
# import seaborn to visualize data that built on top of matplotlib
import seaborn as sns

# import matplotlib
import matplotlib.pyplot as plt

# import pipeline to make machine learning pipeline to overcome data leakage problem
from sklearn.pipeline import Pipeline

# import StandardScaler to Column Standardize the data
# many algorithm assumes data to be Standardized
from sklearn.preprocessing import StandardScaler

# train_test_split is used to split the data into train and test set of given data
from sklearn.model_selection import train_test_split

# KFold is used for defining the no.of folds for Cross Validation
from sklearn.model_selection import KFold

# cross_val_score is used to find the score on given model and KFlod
from sklearn.model_selection import cross_val_score

# used for Hyper-parameter
from sklearn.model_selection import GridSearchCV

# classification report show the classification report
# precision, recall, f1-score
from sklearn.metrics import classification_report

# accuracy score is also a metrics to judge the model mostly used for Balanced dataset
# not better for Imabalanced dataset
from sklearn.metrics import accuracy_score

# confusion matrix show the comparision between actual label and predicted label of data
# mostly used for Binary classification
from sklearn.metrics import confusion_matrix

# importing different algorithms to train our data and find better model among all algorithms
from sklearn.linear_model import LogisticRegression     # Classic Logistic Regression
from sklearn.tree import DecisionTreeClassifier         # Decision Tree
from sklearn.svm import SVC                             # Support Vector Machines
from sklearn.svm import LinearSVC                       # Linear SVM
from sklearn.linear_model import Perceptron             # Perceptron
from sklearn.ensemble import RandomForestClassifier     # Random Forest Classification
from sklearn.linear_model import SGDClassifier          # Stochastic Gradient Classification
from sklearn.ensemble import GradientBoostingClassifier # Gradient Boost Classifier
from sklearn.naive_bayes import GaussianNB              # Gaussian Naive Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# KNN is slow algorithm for runtime b'z it doesn't learn anything at the time of fitting the model
# KNN just stores every datapoint and find K Nearest Neighbor and
# among all nearest neighbor whichever has high no.of points that is the label of that query point
# Because it store every datapoint in memory to predict the label in runtime, It is not better for large data
# KNN is less used in industry because it is not good for Low "latency"(time to predict the label of given query point) systems like SearchEngines
from sklearn.neighbors import KNeighborsClassifier


# Ensembles generally group more than one model to give better model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# ---Finding the directory where the data is available---
for dirname, _, filenames in os.walk('Downloads/Kaggle Competetions/Titanic Competetion'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# ---Importing the Data---
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# ---Analyzing the Data-----
train_data.shape
train_data.head()
train_data.tail()
train_data.describe().transpose()
train_data.std()
train_data.groupby('Survived').size()
train_data.info()
# --------------------------

# ---Visualizing the Data---
sns.heatmap(train_data.isnull())
sns.countplot(x='Survived', data = train_data)
sns.countplot(x='Survived', hue='Sex', data = train_data)
sns.countplot(x='Survived', hue='SibSp',data=train_data)
sns.distplot(train_data['Age'].dropna())
train_data['Age'].hist(bins=30,color='darkred',alpha=0.5)
# --------------------------

# ---DATA CLEANING and PREPROCESSING---

## ---Age is an important factor, hence grouping it broadly to classify the data---
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels = labels)
test_data['AgeGroup'] = pd.cut(test_data["Age"], bins, labels = labels)

## draw a count plot of Age vs. survival
sns.countplot(x="Survived", hue="AgeGroup", data=train_data)

train_data = train_data.drop(['AgeGroup'], axis = 1)
test_data = test_data.drop(['AgeGroup'], axis = 1)
##---------------------------------------------------------------------------------

fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

pClsAge = train_data.groupby('Pclass').mean()

sns.barplot(x='Pclass', y='Age',data=train_data)

#-------------------------------------------------------
# Finding the missing age of some passengers
# Using the PClass Mean Age for this
#-------------------------------------------------------
'''
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    pClsAge = train_data.groupby('Pclass').mean()
    
    if pd.isnull(Age):
        if Pclass == 1:
            return pClsAge.at[1, 'Age']
        elif Pclass == 2:
            return pClsAge.at[2, 'Age']
        else:
            return pClsAge.at[3, 'Age']
    else:
        return Age
    
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)

# we have also to clean the test data
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)
'''
#take the median value for Age feature based on 'Pclass' and 'Title'
train_data['Age'] = train_data.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))
test_data['Age'] = test_data.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))
#--------------------------------------------------------

#-------------------------------------------------------
# Finding the missing FARE of some passengers
# Using the Mean Fare for this
#-------------------------------------------------------
test_data.set_value(test_data[test_data['Fare'].isnull()].index, 'Fare', train_data['Fare'].mean())
#-------------------------------------------------------

#-------------------------------------------------------
# Finding the missing EMBARKED value of some passengers
# Using the MOST Frequent Value for this
#-------------------------------------------------------
#replacing the missing values in the Embarked feature with S
train_data = train_data.fillna({"Embarked": "S"})
#-------------------------------------------------------

#-------------------------------------------------------
# LABELING the SEX and EMBARKED Columns
#-------------------------------------------------------

## For TRAIN Data
sex = pd.get_dummies(train_data['Sex'],drop_first=True) # getting dummy of 'Sex' column
embark = pd.get_dummies(train_data['Embarked'],drop_first=True) # getting dummy of 'Embarked'

## For TEST Data
sex_test = pd.get_dummies(test_data['Sex'],drop_first=True) # getting dummy of 'Sex' column
embark_test = pd.get_dummies(test_data['Embarked'],drop_first=True) # getting dummy of 'Embarked'

#-------------------------------------------------------

#-------------------------------------------------------
# Check for any important column left with any NULL Value
#-------------------------------------------------------
train_data.isnull().sum()
test_data.isnull().sum()
#-------------------------------------------------------

#-------------------------------------------------------
# DROP columns that are not required
# 'Sex', 'Embarked', 'Name','Ticket','Cabin'
#-------------------------------------------------------
# For TRAIN Data
train_data.drop(['Sex','Embarked','Name','Ticket', 'Cabin'],axis=1,inplace=True)

# For TEST Data
test_data.drop(['Sex','Embarked','Name','Ticket', 'Cabin'],axis=1,inplace=True)
#-------------------------------------------------------

#-------------------------------------------------------
# Join the columns with LABEL values - SEX and EMBARK
#-------------------------------------------------------
# for train
train_data = pd.concat([train_data,sex,embark],axis=1)
# for test
test_data = pd.concat([test_data,sex_test,embark_test],axis=1)
#-------------------------------------------------------

#-------------------------------------------------------
# SPLIT the TRAINING data into VALIDATION Data
#-------------------------------------------------------

target = train_data["Survived"]
predictors = train_data.drop(['Survived', 'PassengerId'], axis = 1)

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
#-------------------------------------------------------

#-------------------------------------------------------
# RUN various algorithms one-by-one to find the accuracy
#-------------------------------------------------------

#-------------------------------------------------------
# GAUSSIAN NAIVE BAYES
#-------------------------------------------------------
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
#-------------------------------------------------------

#-------------------------------------------------------
# LOGISTIC REGRESSION
#-------------------------------------------------------
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
#-------------------------------------------------------

#-------------------------------------------------------
# SUPPORT VECTOR MACHINE
#-------------------------------------------------------
svc = SVC(kernel='rbf', random_state=0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#-------------------------------------------------------

#-------------------------------------------------------
# LINEAR SVC
#-------------------------------------------------------
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
#-------------------------------------------------------

#-------------------------------------------------------
# PERCEPTRON
#-------------------------------------------------------
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
#-------------------------------------------------------

#-------------------------------------------------------
# DECISION TREE CLASSIFIER
#-------------------------------------------------------
decisiontree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
#-------------------------------------------------------

#-------------------------------------------------------
# RANDOM FOREST CLASSIFIER
#-------------------------------------------------------
randomforest = RandomForestClassifier(n_estimators=100, max_depth=5, criterion='entropy', random_state=0)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
#-------------------------------------------------------

#-------------------------------------------------------
# K-NEIGHBORS CLASSIFIER
#-------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
#-------------------------------------------------------

#-------------------------------------------------------
# STOCHASTIC GRADIENT CLASSIFIER
#-------------------------------------------------------
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
#-------------------------------------------------------

#-------------------------------------------------------
# GRADIENT BOOST CLASSIFIER
#-------------------------------------------------------
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
#-------------------------------------------------------

#-------------------------------------------------------
# Checking the ACCURACY SCORE
#-------------------------------------------------------
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 
              'Naive Bayes', 'Perceptron', 'Linear SVC', 'Decision Tree', 
              'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, acc_randomforest, acc_gaussian, acc_perceptron,
              acc_linear_svc, acc_decisiontree, acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
#-------------------------------------------------------

#-------------------------------------------------------
# Performing CROSS-VALIDATION
#-------------------------------------------------------
X = train_data.drop(['Survived', 'PassengerId'], axis=1)
Y = train_data["Survived"]

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Support Vector Machines', 'K-Nearst Neighbor', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 'Decision Tree', 
              'Stochastic Gradient Descent', 'Gradient Boosting Classifier']
models=[SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), 
        GaussianNB(), Perceptron(), LinearSVC(), DecisionTreeClassifier(), SGDClassifier(), 
        GradientBoostingClassifier()]

for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2
#-------------------------------------------------------

#-------------------------------------------------------
# Finalizing the Results with GRADIENT BOOST CLASSIFIER
#-------------------------------------------------------

#set ids as PassengerId and predict survival 
ids = test_data['PassengerId']
predictions = gbk.predict(test_data.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
#-------------------------------------------------------

#-------------------------------------------------------
# Writting the output to the CSV File
#-------------------------------------------------------
output.to_csv('submission.csv', index=False)
print("The submission was successfully saved!")
#-------------------------------------------------------
