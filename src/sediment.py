import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import fastdtw as fastdtw
from fastdtw import fastdtw
from savitzky_golay import savitzky_golay

sns.set(rc={'figure.figsize':(11.7,5)})


class sedimentRecord:

    def __init__(self, target, data):
        self.target: pd.Series = target
        self.data: pd.Series = data

    def simple_dtw(self):
        self.distance: float = dtw.distance(self.data, self.target)


if __name__ == "__main__":
    df_1100 = pd.read_csv('data/1100_complete.csv')
    data = df_1100['d18O_pl']
    data = np.array(data)
    data = savitzky_golay(data, 11, 3)
    stack = pd.read_csv('data/LR04stack.txt', sep='\\t', engine='python')
    stack = stack[['Time_ka', 'Benthic_d18O_per-mil']]
    _tmp = stack[stack['Time_ka'] <= 372]
    target = _tmp['Benthic_d18O_per-mil']
    target = zscore(target)   

    record = sedimentRecord(data, target)
    record.simple_dtw()
    print(record.distance)

