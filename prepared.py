"""Read data."""
import os
import subprocess
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from confs import logconf

logger = logconf.Logger(__file__).logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/')


class Prepared():
    """Prepared."""

    events_train = pd.read_csv(os.path.join(DATA_DIR, 'events_train.csv'))
    # labels_train = pd.read_csv(os.path.join(DATA_DIR, 'labels_train.csv'))
    # events_test = pd.read_csv(os.path.join(DATA_DIR, 'events_test.csv'))

    def __init__(self):
        """Init."""
        logger.debug(self.events_train.info())
        logger.debug(self.events_train.head())
        logger.debug(self.events_train.describe())

        # info of events_train
        # self.events_test.hist(bins=50, figsize=(20, 15))
        # plt.savefig('events_test')
        # subprocess.call(['catimg', '-f', 'events_test.png'])

    def train(self):
        """Prepared train data."""
        for idx, t in enumerate(self.events_train['time'][:1000]):
            logger.debug('time {}'.format(idx))
            self.events_train['time'][idx] = datetime.fromtimestamp(t).month

        self.events_train['time'][:1000].hist(bins=50, figsize=(20, 15))
        plt.savefig('events_train')
        subprocess.call(['catimg', '-f', 'events_train.png'])
        self.events_train['title_id'].hist(bins=200, figsize=(20, 15))
        plt.savefig('events_train-title_id')
        subprocess.call(['catimg', '-f', 'events_train-title_id.png'])


if __name__ == '__main__':
    p = Prepared()
    p.train()