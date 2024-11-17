import argparse
import os
import pandas as pd
import pickle
import time
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from model.TLEL import TLEL
import random


def load_csv(file_name):
    if not os.path.exists(file_name):
        assert f"File {file_name} not found"
    df = pd.read_csv(file_name)
    return df


def load_data(train_file, test_file):
    if not os.path.exists(train_file):
        assert f"Dataset not found!"
    # if not os.path.exists(test):
    #     assert f"Folder 'features' not found!"

    train_df = pd.read_json(train_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)

    return train_df, test_df


def get_params():
    parser = argparse.ArgumentParser()
    model_names = ["la", "lr", "tlel", "sim"]
    parser.add_argument("--model", type=str, required=True, choices=model_names)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--save_path", type=str, required=True)
    # parser.add_argument("--prj", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

params = get_params()
random.seed(42)
np.random.seed(42)

# load data
cols = (
    ["la"]
    if params.model == "la"
    else [
        "ns",
        "nd",
        "nf",
        "entropy",
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
train_df, test_df= load_data(params.train_data, params.test_data)
COLS = cols
# train and evaluate model
print("Start training")
for col in COLS:
    cols = [col]
    print(col)
    
    X_train = train_df.loc[:, cols]
    y_train = train_df.loc[:, "label"]
    X_test = test_df.loc[:, cols]
    y_test = test_df.loc[:, "label"]
    id = test_df.loc[:, 'commit_id']
    if params.model == "sim":
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(
            X_train, y_train
        )
        
            
    start = time.time()
    if params.model == "la" or params.model == "lr":
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif params.model == "sim":
        model = RandomForestClassifier(random_state=42)
    elif params.model == "tlel":
        model = TLEL()

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    y_pred = [1 if y >= threshold else 0 for y in y_proba]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prc = precision_score(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    label_df = pd.DataFrame({"commit_hash": id,"label": y_test, "proba": y_proba, "pred": y_pred})
    print(f"auc: {auc:.3f}")
    print(f"acc: {acc:.3f}")
    print(f"f1: {f1:.3f}")
    print(f"prc: {prc:.3f}")
    print(f"rc: {rc:.3f}")

# res_dict = {
#     "commit_hash": id,
#     "label": y_test,
#     "pred": y_proba
# }
# df = pd.DataFrame(res_dict)

# out_json = {
#     "auc": auc,
#     "acc": acc,
#     "f1": f1,
#     "prc": prc,
#     "rc": rc
# }

# print(
#     f"\tFinish training: {round(time.time() - start, 4)} seconds\n\tAUC: {round(auc, 4)}"
# )

# Example
    # corr_cols = cols
    # corr_cols.append('label')
    # correlations = test_df.loc[:, corr_cols].corr()['label'].drop('label')  # Drop 'Label' itself from the correlation
    # print("Correlations with label:")
    # print(correlations)

    # save model
    print("\tSaving model... ", params.model)
    path = os.path.join(params.save_path, params.model)

    if not os.path.exists(path):
        os.makedirs(path)
    save_file = f"{params.model}_{col}_only.pkl"
    with open(os.path.join(path, save_file), "wb") as f:
        pickle.dump(model, f)
    label_df.to_csv(f"{path}/{params.model}_{col}_only_pred_scores.csv", index=False)

# r = permutation_importance(model, X_test, y_test,
#                            n_repeats=30,
#                            random_state=0,
#                            n_jobs=2)

# with open(f"{path}/{params.model}_features_importances.txt", "w") as f:
#     for i in r.importances_mean.argsort()[::-1]:
#         f.write(f"{cols[i]:<8}"
#                 f"{r.importances_mean[i]:.3f}"
#                 f" +/- {r.importances_std[i]:.3f}\n")
        
#         print(f"{cols[i]:<8}"
#                 f"{r.importances_mean[i]:.3f}"
#                 f" +/- {r.importances_std[i]:.3f}")

print("Done!")
