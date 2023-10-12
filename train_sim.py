from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample

import logging
import numpy as np

logging.basicConfig(
    filename="log/sim.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_sim(prj, mode="within", split=None):
    logging.info("Training Sim:")
    logging.info(f"Project: {prj} - Mode: {mode}")
    # load data
    train_f, test_f, train_cc, test_cc = load_data(prj, mode)

    if split is not None:
        logging.info(f"Split data with rate: {split}")
        train_f, valid_f, train_cc, valid_cc = split_data(train_f,
                                                          train_cc,
                                                          rate=split)
    majority_class = train_f[train_f['bug'] == 0]
    minority_class = train_f[train_f['bug'] == 1]

    n_samples = len(minority_class)
    majority_undersampled = resample(majority_class,
                                     replace=False,
                                     n_samples=n_samples,
                                     random_state=42)

    train_f = pd.concat([minority_class, majority_undersampled])
    X_train, y_train = train_f.iloc[:, 5:], train_f.iloc[:, 3]
    X_test, y_test = test_f.iloc[:, 5:], test_f.iloc[:, 3]

    # train
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # evaluate
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    logging.info(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
    logging.info(classification_report(y_test, y_pred))

    if not os.path.exists("sim"):
        os.mkdir("sim")

    file_name = f"sim/{prj}_{mode}.pkl" if split is None else f"sim/{prj}_{mode}_{split}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


for prj in projects:
    train_sim(prj, mode="within")
    train_sim(prj, mode="cross")
    train_sim(prj, mode="within", split=0.8)
    train_sim(prj, mode="cross", split=0.8)

logging.shutdown()