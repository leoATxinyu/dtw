import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from savitzky_golay import savitzky_golay
from scipy.spatial.distance import euclidean
from dtaidistance import dtw, dtw_visualisation


sns.set(rc={'figure.figsize':(11.7,5)})


class sedimentTimeWarp:

    def __init__(self, target: pd.Series, data: pd.Series):
        self.target: pd.Series = target
        self.data: pd.Series = data

    @staticmethod
    def smooth_time_series(time_series: pd.Series, window_size: int = 11, polynomial: int = 3):
        return savitzky_golay(time_series, window_size, polynomial)

    def simple_distance(self, use_smoothed=False, window_size: int = None, polynomial: int = None):
        if use_smoothed:
            if not window_size or not polynomial:
                print('ERROR: Missing window_size or polynomial parameter.')
                return
            self.target = self.smooth_time_series(self.target, window_size, polynomial)
        self.distance: float = dtw.distance(self.data, self.target)
        print(f'Distance: {self.distance}')



if __name__ == "__main__":





    df_1100 = pd.read_csv('data/core_1100.csv')
    data = df_1100['d18O_pl']
    data = np.array(data)
    data = savitzky_golay(data, 11, 3)
    stack = pd.read_csv('data/LR04stack.txt', sep='\\t', engine='python')
    stack = stack[['Time_ka', 'Benthic_d18O_per-mil']]
    _tmp = stack[stack['Time_ka'] <= 372]
    target = _tmp['Benthic_d18O_per-mil']
    target = zscore(target)   

    record = sedimentTimeWarp(data, target)
    record.simple_distance(use_smoothed=True, window_size=11, polynomial=3)
    # print(record.distance)