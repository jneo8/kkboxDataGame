"""Read data."""
import os
import subprocess
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

from confs import logconf

logger = logconf.Logger(__file__).logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/')


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
        self.train = read(name='events_train_v1')
        self.train_x = self.train.drop(self.train.columns[[0]], axis=1)

        self.label = read(name='labels_train')
        self.label_x = self.label['title_id'].values.tolist()

        some = 75315
        self.some_digit = self.train.loc[some]
        self.some_digit_label = self.label.loc[some]

    def main(self):
        """Step."""
        self._svm()
        # logger.debug(self.label_x[:200])

        # logger.debug(self.train_x.iloc(self.num))

    def _svm(self):
        """SVM."""
        clf = svm.SVC()
        clf.fit(self.train_x, self.label_x)

        logger.info(clf.predict(self.some_digit))
        logger.info(self.some_digit_label)

if __name__ == '__main__':
    t = Train()
    t.main()
