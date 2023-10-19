import argparse
import os
import pandas as pd
import pickle
import time

from train_lapredict import *
from train_tlel import *
from train_lrjit import *
from train_sim import *

def load_csv(path, file_name):
    if not os.path.exists(os.path.join(path, file_name)):
        assert f"File {file_name} not found: {path}"
    df = pd.read_csv(os.path.join(path, file_name))
    return df

def load_data(path, prj, part):
    if not os.path.exists(os.path.join(path, prj)):
        assert f"Dataset {prj} not found in {path}!"
    if not os.path.exists(os.path.join(path, prj, 'features')):
        assert f"Folder 'features' not found in {path}/{prj}!"
    
    train_file = f"{prj}_{part}.csv"
    train_df = load_csv(os.path.join(path, prj, 'features'), train_file)
    test_file = f"{prj}_part_5.csv"
    test_df = load_csv(os.path.join(path, prj, 'features'), test_file)
    
    return train_df, test_df


def get_params():
    parser = argparse.ArgumentParser()
    model_names = ["la", "lr", "tlel", "sim"]
    parser.add_argument("--model", type=str, required=True, choices=model_names)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    train_parts = ["part_1_part_4", "part_3_4", "part_4"]
    parser.add_argument("--train_part", type=str, required=True, choices=train_parts)
    parser.add_argument("--prj", type=str, required=True)

params = get_params()

# load data
train_df, test_df = load_data(params.data_path, params.prj, params.train_part)

# train and evaluate model
train_eval = {
    "la": train_eval_la,
    "lr": train_eval_lr,
    "tlel": train_eval_tlel,
    "sim": train_eval_sim
}
print(f"Start trainning {params.model} on {params.prj}:\nTrain: part {params.train_part} - Test: part 5")
start = time.time()
model, auc = train_eval[params.model](train_df, test_df)
print(f"\tFinish training: Time: {(time.time() - start)}\n\tAUC: {auc}")

# save model
print("Saving model... ", end="")
path = os.path.join(params.save_path, params.prj, params.model)
if not os.path.exists(path):
    os.makedirs(path)
save_file = f"{params.prj}_{params.train_part}.pkl"
with open(os.path.join(path, save_file), "wb") as f:
    pickle.dump(model, f)
    
with open(os.path.join(path, f"auc.txt"), "a") as f:
    f.write(f"{params.train_part}: {auc}\n")
print("Done!")


