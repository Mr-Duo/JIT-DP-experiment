from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def train_eval_la(train_df, test_df):
    X_train, y_train = np.array(train_df["la"],
                                dtype=np.float32), np.array(train_df["bug"],
                                                            dtype=np.int16)
    X_test, y_test = np.array(test_df["la"],
                              dtype=np.float32), np.array(test_df["bug"],
                                                          dtype=np.int16)

    # train
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train.reshape(-1, 1), y_train)

    # evaluate
    y_proba = model.predict_proba(X_test.reshape(-1, 1))
    y_pred = model.predict(X_test.reshape(-1, 1))
    auc = roc_auc_score(y_test, y_proba[:,1])
    return model, auc
