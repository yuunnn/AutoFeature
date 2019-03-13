import numpy as np
import pandas as pd
import time
from abc import ABC,abstractmethod
from .utils import fillna
from typing import List

class Propecssing(ABC):

    @abstractmethod
    def fit(self,X):
        pass

    @abstractmethod
    def transform(self,X):
        pass
    
    def type_confirm(self,X):
        if isinstance(X,(pd.DataFrame,pd.Series)):
            self._x_type = 1
            self._x_columns = list(X.columns)
        elif isinstance(X,(np.ndarray,np.array)):
            self._x_type = 0
        else:
            raise Exception("data should be pandas or numpy date type")

    @abstractmethod
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

class TimeSeriesor(Propecssing):
    
    def __init__(self,timeseries_colums : int,target='split',str_format="%Y%m%d",keepdims : int=3,plus_weekday=False):
        '''
        timeseries_colums为需要处理的列，只能一次处理一列，
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

    def fit(self,X):
        raise NotImplementedError('this class does not have the fit or transform function,just use fit_transform')
        
    def transform(self,X):
        raise NotImplementedError('this class does not have the fit or transform function,just use fit_transform')
        
    def fit_transform(self,X):
        self.type_confirm(X)
        
        X = np.array(X)

        timescol = X[:,self._timeseries_colums]
        timescol = [time.strptime('{}'.format(i),self._format) for i in timescol]

        for dims in range(self._keepdims):
            plus_col = [timestruct[dims] for timestruct in timescol]
            X = np.column_stack((X,plus_col))
        if self._plus_weekday == True:
            plus_col = [timestruct.tm_wday + 1 for timestruct in timescol]
            X = np.column_stack((X,plus_col))
        if self._x_type == 1:
            if self._plus_weekday == False:
                self._x_columns.extend(self._struct[:self._keepdims])
                X = pd.DataFrame(X,columns = self._x_columns)

            else:
                self._x_columns.extend(self._struct[:self._keepdims])
                self._x_columns.append('tm_wday')
                X = pd.DataFrame(X,columns = self._x_columns)
        return X

class WindowSlider(Propecssing):

    def fit_transform(self):
        pass

class LaberEncoder(Propecssing):
    '''
    columns为需要labelencoder的特征
    '''    
    def __init__(self,columns : List[int],ifcopy=True):
        self._columns = columns
        self._ifcopy = ifcopy

    def fit(self,X):
        self.type_confirm(X)
        
        self._encdict = {}
        for i in self._columns:
            if self._x_type == 0:
                _class,y = np.unique(X[:,i],return_inverse=True)
            if self._x_type == 1:
                _class,y = np.unique(X.iloc[:,i],return_inverse=True)
            i_encoder = dict(zip(_class,y))
            self._encdict.update({i:i_encoder})
    
    def transform(self,X):
        if self._ifcopy:
            _X = np.array(X)
            for i in self._columns:
                mapping = np.vectorize(lambda v: self._encdict[i][v])
                _X[:,i] = mapping(_X[:,i])
            if self._x_type == 0:
                return _X
            if self._x_type == 1:
                return pd.DataFrame(_X,columns = self._x_columns)
        else:
            for i in self._columns:
                mapping = np.vectorize(lambda v: self._encdict[i][v])
                if self._x_type == 0:
                    X[:,i] = mapping(X[:,i])
                if self._x_type == 1:
                    X.iloc[:,i] = mapping(X.iloc[:,i])
            
    def ger_encoder(self):
        return self._encdict

class Combinator(Propecssing):

    def fit_transform(self):
        pass

