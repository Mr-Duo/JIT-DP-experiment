from utils import *
from model.JITLine import JITLine
from sklearn.metrics import classification_report, roc_auc_score
import logging
import numpy as np

logging.basicConfig(
    filename="jitline.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train_jitline(prj, language, mode="within", split=None):
    logging.info("Training JITLine:")
    logging.info(f"Project: {prj} - Mode: {mode}")
    # load data
    train_f, test_f, train_cc, test_cc = load_data(prj, mode)

    if split is not None:
        logging.info(f"Split data with rate: {split}")
        train_f, valid_f, train_cc, valid_cc = split_data(train_f,
                                                          train_cc,
                                                          rate=split)

    train_code_change = {"commit id": train_cc[0], "code change": train_cc[3]}
    test_code_change = {"commit id": test_cc[0], "code change": test_cc[3]}

    # train
    model = JITLine(language=language)
    model.fit(train_f, train_code_change)

    # evaluate
    y_proba = model.predict_proba(test_f, test_code_change)
    y_pred = [1 if p[1] > 0.5 else 0 for p in y_proba]
    y_test = np.array(test_f['bug'], dtype=np.int8)
    logging.info(f"AUC: {roc_auc_score(y_test, y_proba[:,1])}")
    logging.info(classification_report(y_test, y_pred))

    if not os.path.exists("jitline"):
        os.mkdir("jitline")

    file_name = f"jitline/{prj}_{mode}.pkl" if split is None else f"jitline/{prj}_{mode}_{split}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


prjs = {
    "gerrit": "java",
    "go": "golang",
    "jdt": "java",
    "openstack": "c++",
    "qt": "c++",
    "platform": "java"
}

for prj, lang in prjs.items():
    train_jitline(prj, lang, mode="within")
    train_jitline(prj, lang, mode="cross")
    train_jitline(prj, lang, mode="within", split=0.8)
    train_jitline(prj, lang, mode="cross", split=0.8)

logging.shutdown()