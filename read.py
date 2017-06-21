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


def read():
    """read data info."""
    events_train = pd.read_csv(os.path.join(DATA_DIR, 'events_train_v5.csv'))
    # labels_train = pd.read_csv(os.path.join(DATA_DIR, 'labels_train.csv'))
    # events_test = pd.read_csv(os.path.join(DATA_DIR, 'events_test.csv'))

    logger.debug(events_train.info())
    logger.debug(events_train.head())
    logger.debug(events_train.describe())

    # logger.debug(labels_train.info())
    # logger.debug(labels_train.head())
    # logger.debug(labels_train.describe())

    # logger.debug(events_test.info())
    # logger.debug(events_test.head())
    # logger.debug(events_test.describe())


if __name__ == '__main__':
    read()
