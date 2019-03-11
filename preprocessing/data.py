import numpy as np
import pandas as pd
import time
from datetime import datetime
from abc import ABC,abstractmethod


class Propecssing(ABC):

    @abstractmethod
    def fit_transform(self):
        pass

class TimeSeriesor(Propecssing):
    
    def __init__(self,timeseries_colums,target='split',str_format="%Y%m%d",keepdims=3,plus_weekday=False):
        '''
        keepdims为需要拆分的时间维度，例如keepdims=2，就是将时间序列拆为年和月两列（新增）。
        plus_weekday为是否输出星期几。
        数据只能为numpy或者pandas的数据格式
        '''
        self._timeseries_colums = timeseries_colums
        self._target = target
        self._format = str_format
        self._keepdims = keepdims
        self._plus_weekday = plus_weekday
        self._struct = ('tm_year', 'tm_mon', 'tm_mday', 'tm_hour', 'tm_min', 'tm_sec')
        if isinstance(self._keepdims,int) == False:
            raise ValueError('keepdims should be int ')
        if self._keepdims > len(self._format.split('%')) - 1 :
            raise Exception("keepdims should be smaller than len(str_format)")
        if self._keepdims > 6 or self._keepdims < 1:
            raise ValueError('keepdims should be >1 and <=6 ')

        
    def fit_transform(self,X):
        
        if isinstance(X,pd.DataFrame) or isinstance(X,pd.Series):
            _x_type = 1
            _x_columns = list(X.columns)
        elif isinstance(X,np.ndarray) or isinstance(X,np.array):
            _x_type = 0
        else:
            raise Exception("data should be pandas or numpy date type")
        
        X = np.array(X)
        timescol = X[:,self._timeseries_colums]
        timescol = [time.strptime('{}'.format(i),self._format) for i in timescol]
        
        for dims in range(self._keepdims):
            plus_col = [timestruct[dims] for timestruct in timescol]
            X = np.column_stack((X,plus_col))
        
        if self._plus_weekday == True:
            plus_col = [timestruct.tm_wday + 1 for timestruct in timescol]
            X = np.column_stack((X,plus_col))
            
        if _x_type == 0:
            return X
        if _x_type == 1:
            if self._plus_weekday == False:
                _x_columns.extend(self._struct[:self._keepdims])
                return pd.DataFrame(X,columns = _x_columns)
            else:
                _x_columns.extend(self._struct[:self._keepdims])
                _x_columns.append('tm_wday')
                return pd.DataFrame(X,columns = _x_columns)

class WindowSlider(Propecssing):

    def fit_transform(self):
        pass

class LaberEncoder(Propecssing):
    
    def fit_transform(self):
        pass

class Combinator(Propecssing):

    def fit_transform(self):
        pass

