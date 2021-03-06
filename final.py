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


class Final():
    """Train."""

    def __init__(self):
        """Init."""
        x = read(name='events_test_v8')
        self.x = x
        logger.debug(x.head())

    def mlp(self):
        """Gen predict_data."""
        mlp_clf = joblib.load(os.path.join(PKL_DIR, 'mlp.pkl'))
        uqique_user = (
            pd.read_csv(
                os.path.join(DATA_DIR, 'events_test_v8.csv'),
            )
        )['user_id'].unique()

        p = mlp_clf.predict(self.x)

        logger.debug(len(uqique_user))
        logger.debug(len(p))

        raw_data = {
            'user_id': [],
            'title_id': [],
        }
        for u, t in zip(uqique_user, p):
            logger.debug('{u} {p}'.format(u=u, p=t))
            raw_data['user_id'].append(str(u).zfill(8))
            raw_data['title_id'].append(str(t).zfill(8))

        df = pd.DataFrame(raw_data, columns=['user_id', 'title_id'])
        df.to_csv(os.path.join(DATA_DIR, 'final_mlp.csv'), index=False)

    def rfc(self):
        """Gen predict_data."""
        rfc_clf = joblib.load(os.path.join(PKL_DIR, 'rfc_v8.pkl'))
        uqique_user = (
            pd.read_csv(
                os.path.join(DATA_DIR, 'events_test_v8.csv'),
            )
        )['user_id'].unique()

        p = rfc_clf.predict(self.x)

        logger.debug(len(uqique_user))
        logger.debug(len(p))

        raw_data = {
            'user_id': [],
            'title_id': [],
        }
        for u, t in zip(uqique_user, p):
            logger.debug('{u} {p}'.format(u=u, p=t))
            raw_data['user_id'].append(str(u).zfill(8))
            raw_data['title_id'].append(str(t).zfill(8))

        df = pd.DataFrame(raw_data, columns=['user_id', 'title_id'])
        df.to_csv(os.path.join(DATA_DIR, 'final_rfc_v8.csv'), index=False)


if __name__ == '__main__':
    f = Final()
    f.rfc()
