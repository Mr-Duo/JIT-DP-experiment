from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import logging
import numpy as np

logging.basicConfig(
    filename="log/lapredict.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_la(prj, mode="within", split=None):
    logging.info("Training LApredict:")
    logging.info(f"Project: {prj} - Mode: {mode}")
    # load data
    train_f, test_f, train_cc, test_cc = load_data(prj, mode)

    if split is not None:
        logging.info(f"Split data with rate: {split}")
        train_f, valid_f, train_cc, valid_cc = split_data(train_f,
                                                          train_cc,
                                                          rate=split)

    X_train, y_train = np.array(train_f["la"],
                                dtype=np.float32), np.array(train_f["bug"],
                                                            dtype=np.int16)
    X_test, y_test = np.array(test_f["la"],
                              dtype=np.float32), np.array(test_f["bug"],
                                                          dtype=np.int16)

    # train
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train.reshape(-1, 1), y_train)

    # evaluate
    y_proba = model.predict_proba(X_test.reshape(-1, 1))
    y_pred = model.predict(X_test.reshape(-1, 1))
    logging.info(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
    logging.info(classification_report(y_test, y_pred))

    if not os.path.exists("lapredict"):
        os.mkdir("lapredict")

    file_name = f"lapredict/{prj}_{mode}.pkl" if split is None else f"lapredict/{prj}_{mode}_{split}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


for prj in projects:
    train_la(prj, mode="within")
    train_la(prj, mode="cross")
    train_la(prj, mode="within", split=0.8)
    train_la(prj, mode="cross", split=0.8)

logging.shutdown()