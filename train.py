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


def read(name):
    """Readd origin data."""
    return pd.read_csv(os.path.join(DATA_DIR, '{}.csv'.format(name)))


class Train():
    """Train."""

    def __init__(self):
        """Init."""
        train_x = pd.read_csv(os.path.join(DATA_DIR, 'events_train_v1.csv'))

        # info of events_train
        logger.debug(train_x.head())

if __name__ == '__main__':
    t = Train()
