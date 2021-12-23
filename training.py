import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

model_name = "xgb_model.pkl"

def train():
    df = pd.read_csv('dataset/employee_churn.csv')
    df_new = df.copy()
    df_new.drop(['empid'], axis = 1, inplace = True)
    df_new.dropna(axis = 0, inplace = True)

    salary_encoded = pd.get_dummies(df_new['salary'], drop_first = True)
    df_new = pd.concat([df_new, salary_encoded], axis = 1)
    df_new.drop(['salary'], axis = 1, inplace = True)

    X = df_new.drop(['left'], axis = 1)
    y = df_new['left']

    xgb_clf = XGBClassifier(objective='binary:logistic', learning_rate = 0.1, max_depth = 10, n_estimator = 100)
    xgb_clf.fit(X, y)
    pickle.dump(xgb_clf, open(model_name, "wb"))

train()