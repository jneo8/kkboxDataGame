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

    def main(self):
        """Prepared test data."""
        self.gen(
            origin_data=self.events_train,
            name='events_train_v7'
        )

        self.gen(
            origin_data=self.events_test,
            name='events_test_v7',
        )

    def gen(self, origin_data, name):
        """Prepared train data."""
        uqique_user = origin_data['user_id'].unique()
        uqique_user.sort()

        uqique_title = origin_data['title_id'].unique()
        uqique_title.sort()
        uqique_title = np.array([str(t) for t in uqique_title])

        time_list_1 = []
        for t in uqique_title:
            for m in [6, 7, 8, 9, 10]:
            # for d in range(1, 32):
            # for h in range(0, 24):
                time_list_1.append(
                    '{t}_{m}'
                    .format(t=t, m=m)
                )

        time_list_2 = []
        for m in [6, 7, 8, 9, 10]:
            for d in range(1, 32):
                for h in range(0, 24):
                    time_list_2.append(
                        '{m}_{d}_{h}'
                        .format(m=m, d=d, h=h)
                    )

        base_ = {}

        for t in uqique_title:
            base_.update(
                {
                    t: 0,
                    'times_{}'.format(t): 0,

                }
            )

        for tl in time_list_1:
            base_.update(
                {
                    tl: 0
                }
            )

        for tl in time_list_2:
            base_.update(
                {
                    tl: 0
                }
            )

        for t in uqique_title:
            base_.update(
                {
                    t: 0,
                    'times_{}'.format(t): 0,

                }
            )

        # Start Gen
        uqique_user_len = len(uqique_user)
        len_ = 2 + (2 * len(uqique_title)) + len(time_list_1) + len(time_list_2)
        logger.debug(len_)
        header = ['user_id'] + [str(x) for x in range(0, len_ - 1)]
        with open(
            os.path.join(DATA_DIR, '{name}.csv'.format(name=name)),
            'w',
            encoding='utf-8'
        ) as output:
            output.write(','.join(header) + '\n')

            total = 0
            for idx, u in enumerate(uqique_user):
                list_ = [str(u)]
                data = origin_data[(origin_data.user_id == u)]
                base = base_.copy()

                for index, row in data.iterrows():
                    base[
                        (
                            '{t}_{m}'
                            .format(
                                t=row.title_id,
                                m=datetime.fromtimestamp(row.time).month,
                                # d=datetime.fromtimestamp(row.time).day,
                                # h=datetime.fromtimestamp(row.time).hour,
                            )
                        )
                    ] += 1

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
                    total += row.watch_time
                    base[str(row.title_id)] += row.watch_time
                    base['times_{}'.format(t)] += 1
                for t in uqique_title:
                    list_.append(str(base[t]))
                    list_.append(
                        str(base['times_{}'.format(t)])
                    )
                for tl in time_list_1:
                    list_.append(str(base[tl]))
                for tl in time_list_2:
                    list_.append(str(base[tl]))

                avg_watch_time = total / len(data)
                list_.append(str(avg_watch_time))
                logger.debug(
                    '{name} {total} : {idx}'
                    .format(
                        name=name,
                        total=uqique_user_len,
                        idx=idx
                    )
                )

                output.write(','.join(list_) + '\n')

            # raw_data['user_id'].append(u)
            # for t in uqique_title:
            #     raw_data[t].append(base[t])
            #     raw_data['times_{}'.format(t)].append(
            #         base['times_{}'.format(t)]
            #     )
            # for tl in time_list_1:
            #     raw_data[tl].append(base[tl])
            # for tl in time_list_2:
            #     raw_data[tl].append(base[tl])
            # logger.debug(
            #     '{name} {idx} : {len}'
            #     .format(name=name, idx=idx, len=len_)
            # )
        # columns_ = (
        #     ['user_id'] +
        #     [t for t in uqique_title] +
        #     ['times_{}'.format(t) for t in uqique_title] +
        #     [tl for tl in time_list_1] +
        #     [tl for tl in time_list_2]
        # )
        # df = pd.DataFrame(raw_data, columns=columns_)
        # df.to_csv(os.path.join(DATA_DIR, '{name}.csv'.format(name=name)))

if __name__ == '__main__':
    p = Prepared()
    p.main()
