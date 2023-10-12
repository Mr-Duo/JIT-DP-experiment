from utils import *
from model.TLEL import TLEL
from sklearn.metrics import classification_report, roc_auc_score
import logging
import numpy as np

logging.basicConfig(
    filename="log/tlel.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_tlel(prj, mode="within", split=None):
    logging.info("Training TLEL:")
    logging.info(f"Project: {prj} - Mode: {mode}")
    # load data
    train_f, test_f, train_cc, _ = load_data(prj, mode)

    if split is not None:
        logging.info(f"Split data with rate: {split}")
        train_f, valid_f, _, _ = split_data(train_f, train_cc, rate=split)

    X_train, y_train = train_f.iloc[:, 5:], train_f.iloc[:, 3]
    X_test, y_test = test_f.iloc[:, 5:], test_f.iloc[:, 3]

    # train
    model = TLEL(n_learner=10, n_tree=10)
    model.fit(X_train, y_train)

    # evaluate
    y_proba = model.predict_proba(X_test)
    y_pred = [1 if p[1] > 0.5 else 0 for p in y_proba]
    logging.info(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
    logging.info(classification_report(y_test, y_pred))

    if not os.path.exists("tlel"):
        os.mkdir("tlel")

    file_name = f"tlel/{prj}_{mode}.pkl" if split is None else f"tlel/{prj}_{mode}_{split}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


for prj in projects:
    train_tlel(prj, mode="within")
    train_tlel(prj, mode="cross")
    train_tlel(prj, mode="within", split=0.8)
    train_tlel(prj, mode="cross", split=0.8)

logging.shutdown()