from model.TLEL import TLEL
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np


def train_eval_tlel(train_df, test_df):
    columns = [
        "ns", "nd", "nf", "entrophy", "la", "ld", "lt", "fix", "ndev", "age",
        "nuc", "exp", "rexp", "sexp"
    ]
    X_train, y_train = train_df.loc[columns], np.array(train_df["bug"],
                                                       dtype=np.int16)
    X_test, y_test = test_df.loc[columns], np.array(test_df["bug"],
                                                    dtype=np.int16)

    # train
    model = TLEL(n_learner=10, n_tree=10)
    model.fit(X_train, y_train)

    # evaluate
    y_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_proba)

    return model, auc
