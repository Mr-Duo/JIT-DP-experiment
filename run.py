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


def load_csv(path, file_name):
    if not os.path.exists(os.path.join(path, file_name)):
        assert f"File {file_name} not found: {path}"
    df = pd.read_csv(os.path.join(path, file_name))
    return df


def load_data(path, prj, part):
    if not os.path.exists(os.path.join(path, prj)):
        assert f"Dataset {prj} not found in {path}!"
    if not os.path.exists(os.path.join(path, prj, "features")):
        assert f"Folder 'features' not found in {path}/{prj}!"

    train_file = f"{prj}_{part}.csv"
    train_df = load_csv(os.path.join(path, prj, "features"), train_file)
    test_file = f"{prj}_part_5.csv"
    test_df = load_csv(os.path.join(path, prj, "features"), test_file)

    return train_df, test_df


def get_params():
    parser = argparse.ArgumentParser()
    model_names = ["la", "lr", "tlel", "sim"]
    parser.add_argument("--model", type=str, required=True, choices=model_names)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    train_parts = ["part_1_part_4", "part_3_part_4", "part_4"]
    parser.add_argument("--train_part", type=str, required=True, choices=train_parts)
    parser.add_argument("--prj", type=str, required=True)

    return parser.parse_args()

random.seed(42)
params = get_params()

# load data
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
train_df, test_df = load_data(params.data_path, params.prj, params.train_part)
X_train = train_df.loc[:, cols]
y_train = train_df.loc[:, "bug"]
X_test = test_df.loc[:, cols]
y_test = test_df.loc[:, "bug"]

if params.model == "sim":
    X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(
        X_train, y_train
    )

# train and evaluate model
print(
    f"{params.prj}\n\tStart trainning {params.model}\n\tTrain: {params.train_part} - Test: part_5"
)
start = time.time()
if params.model == "la" or params.model == "lr":
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
elif params.model == "sim":
    model = RandomForestClassifier()
elif params.model == "tlel":
    model = TLEL()

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
label_df = pd.DataFrame({"bug": y_test, "pred": y_proba})

print(
    f"\tFinish training: {round(time.time() - start, 4)} seconds\n\tAUC: {round(auc, 4)}"
)

# save model
print("\tSaving model... ", end="")
path = os.path.join(params.save_path, params.prj, params.model)
if not os.path.exists(path):
    os.makedirs(path)
save_file = f"{params.prj}_{params.train_part}.pkl"
with open(os.path.join(path, save_file), "wb") as f:
    pickle.dump(model, f)

if not os.path.exists(os.path.join(path, "pred_score")):
    os.makedirs(os.path.join(path, "pred_score"))
label_df.to_csv(
    os.path.join(path, "pred_score", f"test_sim_{params.prj}_{params.train_part}.csv")
)

with open(os.path.join(path, f"auc.txt"), "a") as f:
    f.write(f"{params.prj} - {params.model} - {params.train_part}: {auc}\n")
print("Done!")
