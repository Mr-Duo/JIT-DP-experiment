import os, pickle, pandas as pd

data_path = '/Users/nguyenphucthanh/Downloads/data'
projects = ["gerrit", 'go', 'jdt', "openstack", "platform", 'qt']


def load_data(prj, mode):
    path = os.path.join(data_path, prj)
    if mode == "within":
        f_path = os.path.join(path, f'{prj}_k_feature.csv')
        feature_df = pd.read_csv(f_path)
        idx = int(len(feature_df) * 0.8)
        train_f = feature_df[:idx]
        test_f = feature_df[idx:]
    elif mode == "cross":
        path = os.path.join(path, "cross")
        train_f_path = os.path.join(path, 'k_train.csv')
        test_f_path = os.path.join(path, 'k_test.csv')
        train_f = pd.read_csv(train_f_path)
        test_f = pd.read_csv(test_f_path)
    else:
        raise ValueError("Mode must be within or cross")

    cc_path = os.path.join(path, "cc2vec")
    train_cc_path = os.path.join(cc_path, f"{prj}_train.pkl")
    test_cc_path = os.path.join(cc_path, f"{prj}_test.pkl")
    with open(train_cc_path, 'rb') as f:
        train_cc = pickle.load(f)
    with open(test_cc_path, 'rb') as f:
        test_cc = pickle.load(f)
    return train_f, test_f, train_cc, test_cc


def split_data(train_f, train_cc, rate=0.8):
    if len(train_f) != len(train_cc[0]):
        raise ValueError("Length of feature and code change must be equal")
    split_idx = int(len(train_cc[0]) * rate)
    valid_f = train_f[split_idx:]
    train_f = train_f[:split_idx]

    valid_cc = [
        train_cc[0][split_idx:], train_cc[1][split_idx:],
        train_cc[2][split_idx:], train_cc[3][split_idx:]
    ]
    train_cc = [
        train_cc[0][:split_idx], train_cc[1][:split_idx],
        train_cc[2][:split_idx], train_cc[3][:split_idx]
    ]
    return train_f, valid_f, train_cc, valid_cc