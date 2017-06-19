"""Read data."""
import os
import subprocess
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from confs import logconf

logger = logconf.Logger(__file__).logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/')


class Prepared():
    """Prepared."""

    events_train = pd.read_csv(os.path.join(DATA_DIR, 'events_train.csv'))
    # labels_train = pd.read_csv(os.path.join(DATA_DIR, 'labels_train.csv'))
    events_test = pd.read_csv(os.path.join(DATA_DIR, 'events_test.csv'))

    def __init__(self):
        """Init."""
        # logger.debug(self.events_train.head())
        # logger.debug(self.events_train.describe())

        # info of events_train
        # self.events_test.hist(bins=50, figsize=(20, 15))
        # plt.savefig('events_test')
        # subprocess.call(['catimg', '-f', 'events_test.png'])

    def time_to_month(self):
        """Turn time from sec to month."""
        for idx, t in enumerate(self.events_train['time']):
            # logger.debug('time {}'.format(idx))
            self.events_train['time'][idx] = datetime.fromtimestamp(t).month

        self.events_train.to_csv(
            os.path.join(DATA_DIR, 'events_train_v1.csv')
        )

    def test(self):
        """Prepared test data."""
        # info of event_test.csv
        logger.debug(self.events_test.head())
        self.events_test.hist(bins=50, figsize=(20, 15))
        plt.savefig('events_test')
        subprocess.call(['catimg', '-f', 'events_test.png'])

        user_list = self.events_test['user_id'].unique()

        # logger.debug(type(self.events_test))

        # logger.debug(dir(self.events_test))

        raw_data = {'user_id': [], 'title_id': []}
        for idx, u in enumerate(user_list):
            data = self.events_test[(self.events_test.user_id == u)]
            dict_ = {}
            for index, row in data.iterrows():
                if dict_.get(row.title_id):
                    dict_[row.title_id] += row.watch_time
                else:
                    dict_[row.title_id] = row.watch_time
            key = max(dict_, key=dict_.get)
            raw_data['user_id'].append(u)
            raw_data['title_id'].append(key)
            logger.debug('test csv {}'.format(idx))

        df = pd.DataFrame(raw_data, columns=['user_id', 'title_id'])
        df.to_csv(os.path.join(DATA_DIR, 'test_result.csv'))

    def train(self):
        """Prepared train data."""
        uqique_title = self.events_train['title_id'].unique()
        uqique_title.sort()
        uqique_title = np.array([str(t) for t in uqique_title])
        uqique_user = self.events_train['user_id'].unique()
        uqique_user.sort()

        raw_data = {
            t: []
            for t in uqique_title
        }
        raw_data.update({'user_id': []})
        for idx, u in enumerate(uqique_user):
            data = self.events_train[(self.events_train.user_id == u)]
            base = {
                t: 0
                for t in uqique_title
            }
            for index, row in data.iterrows():
                base[str(row.title_id)] += row.watch_time
            logger.debug(base)
            raw_data['user_id'].append(u)
            for t in uqique_title:
                raw_data[t].append(base[t])
            logger.debug(idx)
        logger.debug(raw_data)
        columns_ = [t for t in uqique_title]
        columns_ = ['user_id'] + columns_
        df = pd.DataFrame(raw_data, columns=columns_)
        df.to_csv(os.path.join(DATA_DIR, 'events_train_v1.csv'))

if __name__ == '__main__':
    p = Prepared()
    p.train()
