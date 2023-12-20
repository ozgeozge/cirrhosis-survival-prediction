#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from imblearn.over_sampling import ADASYN


# parameters

model_name = "RF-ADASYN"
output_file = f'model_{model_name}.bin'

# data preparation

df = pd.read_csv("cirrhosis.csv")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.Status.values
y_test = df_test.Status.values

del df_full_train['Status']
del df_test['Status']


numerical_columns = df_full_train.select_dtypes(include=['int64','float64']).columns
categorical_columns = df_full_train.select_dtypes(include=['object']).columns

# KNN Imputation for Numerical Columns
knn_imputer = KNNImputer()
df_full_train[numerical_columns] = knn_imputer.fit_transform(df_full_train[numerical_columns])
df_test[numerical_columns] = knn_imputer.transform(df_test[numerical_columns])


# Simple Imputation for Categorical Columns (replace with the mode)
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_full_train[categorical_columns] = categorical_imputer.fit_transform(df_full_train[categorical_columns])
df_test[categorical_columns] = categorical_imputer.transform(df_test[categorical_columns])

# Encoding the categorical columns by dictionary vectorizer
dv= DictVectorizer(sparse=False)
train_dicts= df_full_train.to_dict(orient='records')
X_full_train =dv.fit_transform(train_dicts)

test_dicts= df_test.to_dict(orient='records')

X_test =dv.transform(test_dicts)

# Oversample training set by using ADASYN method

X_full_train, y_full_train= ADASYN().fit_resample(X_full_train, y_full_train)


# training the final model

print('training the final model')

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1 ,random_state=1)
rf.fit(X_full_train,y_full_train)
y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred,  average="weighted")

print(f'f1_score={f1}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')