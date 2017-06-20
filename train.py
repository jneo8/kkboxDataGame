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
    return pd.read_csv(os.path.join(DATA_DIR, '{}.csv'.format(name)))


class Train():
    """Train."""

    def __init__(self):
        """Init."""
        # train_x = pd.read_csv(os.path.join(DATA_DIR, 'events_train_v1.csv'))
        train_x = read(name='events_train_v1')
        self.train_x = train_x.drop(train_x.columns[[0, 1]], axis=1)

        label_x = read(name='labels_train')
        self.label_x = label_x['title_id'].values.tolist()

        self.num = 10000

    def main(self):
        """Step."""
        self._svm()
        # logger.debug(self.label_x[:200])

    def _svm(self):
        """SVM."""
        clf = svm.SVC()
        clf.fit(self.train_x, self.label_x)

if __name__ == '__main__':
    t = Train()
    t.main()
