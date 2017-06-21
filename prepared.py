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
        pass

    def test(self):
        """Prepared test data."""
        # self.gen(
        #     origin_data=self.events_train,
        #     name='events_train_v3'
        # )

        self.gen(
            origin_data=self.events_test,
            name='events_test_v3',
        )

    def gen(self, origin_data, name):
        """Prepared train data."""
        uqique_user = origin_data['user_id'].unique()[:100]
        uqique_user.sort()

        uqique_title = origin_data['title_id'].unique()
        uqique_title.sort()
        uqique_title = np.array([str(t) for t in uqique_title])

        raw_data = {}

        len_ = len(uqique_user)
        for t in uqique_title:
            raw_data.update(
                {
                    t: [],
                    'times_{}'.format(t): [],
                }
            )

        raw_data.update({'user_id': []})

        time_list = []
        for m in [6, 7, 8, 9, 10]:
            for d in range(1, 32):
                for h in range(0, 24):
                    time_list.append('{m}_{d}_{h}'.format(m=m, d=d, h=h))

        for tl in time_list:
            raw_data.update(
                {
                    tl: []
                }
            )

        # Start Gen
        for idx, u in enumerate(uqique_user):
            data = origin_data[(self.events_train.user_id == u)]
            base = {}

            for t in uqique_title:
                base.update(
                    {
                        t: 0,
                        'times_{}'.format(t): 0,

                    }
                )

            for tl in time_list:
                base.update(
                    {
                        tl: 0
                    }
                )

            for index, row in data.iterrows():
                base[
                    (
                        '{m}_{d}_{h}'
                        .format(
                            m=datetime.fromtimestamp(row.time).month,
                            d=datetime.fromtimestamp(row.time).day,
                            h=datetime.fromtimestamp(row.time).hour,
                        )
                    )
                ] += 1

                base[str(row.title_id)] += row.watch_time
                base['times_{}'.format(t)] += 1
            raw_data['user_id'].append(u)
            for t in uqique_title:
                raw_data[t].append(base[t])
                raw_data['times_{}'.format(t)].append(
                    base['times_{}'.format(t)]
                )
            for tl in time_list:
                raw_data[tl].append(base[tl])
            logger.debug('{idx} : {len}'.format(idx=idx, len=len_))
        columns_ = (
            ['user_id'] +
            [t for t in uqique_title] +
            ['times_{}'.format(t) for t in uqique_title] +
            [tl for tl in time_list]
        )
        df = pd.DataFrame(raw_data, columns=columns_)
        df.to_csv(os.path.join(DATA_DIR, '{name}.csv'.format(name=name)))

if __name__ == '__main__':
    p = Prepared()
    p.train()
