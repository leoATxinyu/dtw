import numpy as np
import pandas as pd
from typing import Union 
from plot_time_warp import *
from scipy.stats import zscore
from savitzky_golay import savitzky_golay
from scipy.spatial.distance import euclidean
from dtaidistance import dtw, dtw_visualisation


class SedimentTimeWarp:

    def __init__(self, target: pd.DataFrame, data: pd.DataFrame):

        if not isinstance(target, pd.DataFrame):
            raise TypeError('Target needs to be of type pandas.DataFrame')
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data needs to be of type pandas.DataFrame')

        self.target: pd.DataFrame = target        
        self.data: pd.DataFrame = data


    @staticmethod
    def smooth_time_series(time_series: Union[pd.core.series.Series, list], window_size: int = 11, polynomial: int = 3):
        """Smooth a time-series using Savitzky-Golay smoothing algorithm
        """        
        if isinstance(time_series, (pd.core.series.Series, np.array, list)):
            return savitzky_golay(time_series, window_size, polynomial)
        else:
            raise TypeError('Time series needs to be of type pd.Series, np.array or list')


    def simple_distance(self, target_col_name: str, data_col_name: str, 
                        target_max_age: int = None, use_smoothed: bool = False, 
                        window_size: int = None, polynomial: int = None):
        """Calculate distance for a given target/data pair
        """

        if use_smoothed:
            if not window_size or not polynomial:
                print("ERROR: Missing 'window_size' or 'polynomial' parameter.")
                return
            self.data[data_col_name] = self.smooth_time_series(self.data[data_col_name], window_size, polynomial)

        if target_max_age:
            target = target['']
        print(self.data[data_col_name])

        print(self.target[target_col_name])
        
        self.distance: float = dtw.distance(self.data[data_col_name], self.target[target_col_name])
        print(f'Distance: {self.distance}')


    def minimize_distance(self, start_time: int, end_time: int, step_size: int):
        """Find the simple minimum distance(s) for a given target/data pair

        Parameters
        ----------
        start_time: int
        end_time: int
        step_size: int 

        Returns
        -------
        

        Example usage
        -------------
        
        """
        min_distances: dict = {}
        for i in range(start_time, end_time, step_size):
            target = self.target[stack['Time_ka'] <= i]
            target = target['Benthic_d18O_per-mil']
            target = zscore(target)
            data = zscore(self.data['d18O_pl'])
            distance = dtw.distance(data, target)
            min_distances[i] = distance
        min_distance: float = min(min_distances.values())
        self.target_time: list = [k for k, v in min_distances.items() if v==min_distance]
        print(f'Minimum distance: ~{round(min_distance, 2)} at target x (time) = {self.target_time}')
        return min_distances


    def get_warping_path(self, target_time: int = None):
        if not target_time and self.target_time:
            target_time = self.target_time
        elif not target_time and not self.target_time:
            raise TypeError("No target time defined. Please run method 'minimize distance' first or add as argument to method call")


    def monte_carlo(self):
        """Perform complex Monte Carlo simulation of various 
        parameters to find minimum distance(s) for a given 
        target/data pair.        
        """
        pass

    def process_data(self):
        pass

    def process_target(self):
        pass



if __name__ == "__main__":

    df_1100 = pd.read_csv('data/core_1100.csv')
    stack = pd.read_csv('data/LR04stack.txt', sep='\\t', engine='python')       
    test_dtw = SedimentTimeWarp(stack, df_1100)
    test_dtw.simple_distance(target_col_name='Benthic_d18O_per-mil', data_col_name='d18O_pl', use_smoothed=True, window_size=11, polynomial=3)
    # distances = test_dtw.minimize_distance(start_time=100, end_time=1000, step_size=100)
    # path = test_dtw.get_warping_path()
    # print(distances.values())





    # df_1100 = pd.read_csv('data/core_1100.csv')
    # data = df_1100['d18O_pl']
    # data = np.array(data)
    # data = savitzky_golay(data, 11, 3)
    # stack = pd.read_csv('data/LR04stack.txt', sep='\\t', engine='python')
    # stack = stack[['Time_ka', 'Benthic_d18O_per-mil']]
    # _tmp = stack[stack['Time_ka'] <= 372]
    # target = _tmp['Benthic_d18O_per-mil']
    # target = zscore(target)   

    # record = sediTimeWarp(data, target)
    # record.simple_distance(use_smoothed=False)
    # print(record.distance)