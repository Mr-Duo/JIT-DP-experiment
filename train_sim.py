from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import numpy as np
import pandas as pd

def train_eval_sim(train_df, test_df):
    majority_class = train_df[train_df['bug'] == 0]
    minority_class = train_df[train_df['bug'] == 1]

    n_samples = len(minority_class)
    majority_undersampled = resample(majority_class,
                                     replace=False,
                                     n_samples=n_samples,
                                     random_state=42)

    train_df = pd.concat([minority_class, majority_undersampled])
    columns = [
        "ns", "nd", "nf", "entrophy", "la", "ld", "lt", "fix", "ndev", "age",
        "nuc", "exp", "rexp", "sexp"
    ]
    X_train, y_train = train_df.loc[columns], np.array(train_df["bug"],
                                                       dtype=np.int16)
    X_test, y_test = test_df.loc[columns], np.array(test_df["bug"],
                                                    dtype=np.int16)

    # train
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # evaluate
    y_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_proba[:,1])

    return model, auc
