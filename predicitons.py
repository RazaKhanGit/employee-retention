import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

model_name = "xgb_model.pkl"

def make_predictions(predict_data):
    xgb_model_loaded = pickle.load(open(model_name, "rb"))
    X_predict = np.array(predict_data)
    X_predict = np.reshape(X_predict, (1, 9))
    return xgb_model_loaded.predict(X_predict)
