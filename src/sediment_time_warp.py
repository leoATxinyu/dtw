import numpy as np
import pandas as pd
from typing import Union 
from plot_time_warp import *
from scipy.stats import zscore
from savitzky_golay import savitzky_golay
from scipy.spatial.distance import euclidean
from dtaidistance import dtw, dtw_visualisation


class SedimentTimeWarp:

    def __init__(self, target: pd.DataFrame, data: Union[pd.Series, np.array, list]):
        self.target: pd.DataFrame = target        
        self.data: Union[pd.Series, np.array, list] = data


    @staticmethod
    def smooth_time_series(time_series: Union[pd.Series, list, np.array], 
                            window_size: int = 11, polynomial: int = 3):
        """Smooth a time-series using Savitzky-Golay smoothing algorithm
        """        
        return savitzky_golay(time_series, window_size, polynomial)
        

    def simple_distance(self, use_smoothed: bool = False, window_size: int = None, polynomial: int = None):
        """Calculate Euclidian distance for a given target/data pair
        """
        if use_smoothed:
            if not window_size or not polynomial:
                print("ERROR: Missing 'window_size' or 'polynomial' parameter.")
                return
            data = self.smooth_time_series(self.data, window_size, polynomial)
        distance: float = dtw.distance(data, self.target.iloc[:,1])
        print(f'Distance: {distance}')
        return distance


    def minimize_distance(self, start_time: Union[int, float], 
                          end_time: Union[int, float], time_step_size: Union[int, float]):
        """Find the minimum Euclidian distance(s) for a given target/data pair by stepping
        through the range of the target series given by [start_time: <time_step_size> :end_time].

        Parameters
        ----------
        start_time: int or float
            The minimum range value to filter the target time-series by (0 > start_time). 
            Needs to be larger than the time_step_size, and larger than 0.

        end_time: int or float
            The maximum range value to filter the target time-series by 0 > end_time). 
            Needs to be larger than the start_time, time_step_size, and larger than 0.

        time_step_size: int or float
            The step size used to filter the target time-series, iterating through the target from start_time to
            end_time in steps=time_step_size.

        Returns
        -------
        distance: float
            The smallest distance found in the search.

        target_time: list[float]
            A list of time associated with the distance variable.

        min_distances: dict
            A dictionary containing time_step:distance pairs, the raw 
            data from which both distance and target_time where selected.

        Example usage
        -------------
        
        """
        data = zscore(self.data)
        target = self.target        
        target.iloc[:,1] = zscore(target.iloc[:,1])
        min_distances: dict = {}

        for i in range(start_time, end_time, time_step_size):
            _target = target[target.iloc[:,0] <= i]
            distance = dtw.distance(data, _target.iloc[:,1])
            min_distances[i] = distance

        distance: float = min(min_distances.values())
        target_time: list[float] = [k for k, v in min_distances.items() if v==distance]
        print(f'Minimum distance found: ~{round(distance, 2)} at time_step_size={target_time}')
        return distance, target_time, min_distances


    def get_warping_path(self, target_time: int = None):
        if not target_time and self.target_time:
            target_time = self.target_time
        elif not target_time and not self.target_time:
            raise TypeError("No target time defined. Please run method 'minimize distance' first or add as argument to method call")


    def monte_carlo(self):
        """Perform complex Monte Carlo simulation of various 
        parameters to find minimum Euclidian distance(s) for 
        given target/data pair.        
        """
        pass

    def process_data(self):
        pass

    def process_target(self):
        pass



if __name__ == "__main__":

    data = pd.read_csv('data/core_1100.csv')
    data = data.iloc[:,1]
    target = pd.read_csv('data/LR04stack.txt', sep='\\t', engine='python') 

    test_dtw = SedimentTimeWarp(target=target, data=data)
    simple_distance = test_dtw.simple_distance(use_smoothed=True, window_size=11, polynomial=3)    
    distance, target_time, _ = test_dtw.minimize_distance(start_time=100, end_time=1000, time_step_size=10)

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