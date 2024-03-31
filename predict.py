import argparse
import os
import pandas as pd
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from model.TLEL import TLEL
import random

def get_params():
    parser = argparse.ArgumentParser()
    model_names = ["la", "lr", "tlel", "sim"]
    parser.add_argument("--model", type=str, required=True, choices=model_names)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--prj", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    
    return parser.parse_args()

random.seed(42)
params = get_params()

def load_csv(file_name):
    if not os.path.exists(file_name):
        assert f"File {file_name} not found"
    df = pd.read_csv(file_name)
    return df

def load_data(test):
    if not os.path.exists(test):
        assert f"Folder 'features' not found!"

    test_file = test
    test_df = load_csv(test_file)

    return train_df, test_df

def get_model(params):
    path = os.path.join(params.save_path, params.prj, params.model, f"{params.prj}.pkl")
    if not os.path.exists:
        print('Model not found')
        exit(1)
    
    model = pickle.load(path)
    return model

def predict(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    label_df = pd.DataFrame({"commit_hash": id,"label": y_test, "pred": y_proba})

    if not os.path.exists(os.path.join(path, "pred_score")):
        os.makedirs(os.path.join(path, "pred_score"))

    label_df.to_csv(
        os.path.join(path, "pred_score", f"{params.name}_{params.model}_{params.prj}.csv"), index=False, sep=','
    )

cols = (
    ["la"]
    if params.model == "la"
    else [
        "ns",
        "nd",
        "nf",
        "entrophy",
        "la",
        "ld",
        "lt",
        "fix",
        "ndev",
        "age",
        "nuc",
        "exp",
        "rexp",
        "sexp",
    ]
)
test_df = load_data(params.test_data)
X_test = test_df.loc[:, cols]
y_test = test_df.loc[:, "bug"]
id = test_df.loc[:, '_id']

model = get_model(params)
predict(model, X_test, y_test)
