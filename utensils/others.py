import time
import numpy as np


def calc_cost(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        f(*args, **kwargs)
        end_time = time.time()
        cost = end_time - start_time
        hours, remainder = divmod(cost, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours = str(int(hours)).zfill(2)
        minutes = str(int(minutes)).zfill(2)
        seconds = str(int(seconds)).zfill(2)
        print(f'cost time: {hours}:{minutes}:{seconds}')

    return wrapper


def get_indices(array1, array2):
    n_points = array2.shape[0]
    indices = np.zeros(n_points)
    for i in range(n_points):
        indices[i] = np.where(np.all(array1 == array2[i, :], axis=1))[0]
    return indices
