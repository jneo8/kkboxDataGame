"""Read data."""
import os
import subprocess
from datetime import datetime

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

from confs import logconf

logger = logconf.Logger(__file__).logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/')
PKL_DIR = os.path.join(BASE_DIR, 'pkls/')


def read(name):
    """Readd origin data."""
    return pd.read_csv(
        os.path.join(DATA_DIR, '{}.csv'.format(name)),
        index_col=['user_id'],
    )


class Train():
    """Train."""

    def __init__(self):
        """Init."""
        # train_x = pd.read_csv(os.path.join(DATA_DIR, 'events_train_v1.csv'))

        limit = 500
        # limit = 62307

        x = read(name='events_train_v1')
        logger.debug(x.head())
        x = (
            x.drop(x.columns[[0]], axis=1)
            .iloc[0:limit]
        )
        y = read(name='labels_train')
        y = y['title_id'].values.tolist()[:limit]
        # y = y['title_id'].values.tolist()

        test_size = 0.2
        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(
                x, y, test_size=test_size, random_state=42
            )
        )

    def main(self):
        """Step."""
        # self._svm()
        self._mlp()

    def _svm(self):
        """SVM."""
        svc_clf = svm.SVC()
        logger.debug('Start svc_clf.fit')
        svc_clf.fit(self.x_train, self.y_train)

        logger.debug('cross_val_score')
        scores = cross_val_score(
            svc_clf,
            self.x_test,
            self.y_test,
            cv=3,
            scoring="accuracy",
        )
        logger.debug(scores)

    def _mlp(self):
        """MLP."""
        mlp_clf = MLPClassifier()
        mlp_clf.fit(self.x_train, self.y_train)

        scores = cross_val_score(
            mlp_clf,
            self.x_test,
            self.y_test,
            cv=3,
            scoring="accuracy",
        )
        logger.info(scores)
        joblib.dump(mlp_clf, 'mlp.pkl')
        

if __name__ == '__main__':
    t = Train()
    t.main()
