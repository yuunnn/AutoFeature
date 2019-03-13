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
    
    def __init__(self,timeseries_colums,target='split',str_format="%Y%m%d",keepdims : int=3,plus_weekday=False,
                     plus_yearday=False,timethrough : int=0):
        '''
        timeseries_colums为需要处理的列，只能一次处理一列，
        keepdims为需要拆分的时间维度，例如keepdims=2，就是将时间序列拆为年和月两列（新增）。
        plus_weekday为是否输出星期几。
        pus_yearday为是否输出在一年中的多少天
        timethrough为输出的时候按照keepdim的单位进行时间增减，例如keepdim=2，timethrough=-1，则输出的时候月份减少一月。
        数据只能为numpy或者pandas的数据格式
        '''
        self._timeseries_colums = timeseries_colums
        self._target = target
        self._format = str_format
        self._keepdims = keepdims
        self._plus_weekday = plus_weekday
        self._plus_yearday = plus_yearday
        self._timethrough = timethrough
        self._struct = ('tm_year{}'.format(self._timeseries_colums), 
                            'tm_mon{}'.format(self._timeseries_colums),
                            'tm_mday{}'.format(self._timeseries_colums),
                            'tm_hour{}'.format(self._timeseries_colums),
                            'tm_min{}'.format(self._timeseries_colums),
                            'tm_sec{}'.format(self._timeseries_colums))
        
        if isinstance(self._keepdims,int) == False:
            raise ValueError('keepdims should be int ')
        if self._keepdims > len(self._format.split('%')) - 1 :
            raise Exception("keepdims should be smaller than len(str_format)")
        if self._keepdims > 6 or self._keepdims < 1:
            raise ValueError('keepdims should be >1 and <=6 ')
        if (self._plus_weekday  or self._plus_weekday) and len(self._format.split('%')) - 1  <3:
            raise ValueError('date data did not contain days')

    def fit(self,X):
        raise NotImplementedError('this class does not have the fit or transform function,just use fit_transform')
        
    def transform(self,X):
        raise NotImplementedError('this class does not have the fit or transform function,just use fit_transform')
    
    def to_timethouth(self,_time):
        time_stamp = time.mktime(_time)
        
        if self._keepdims in (3,4,5,6):
            day_info = int(self._keepdims == 3) * 24 * 60 * 60
            hour_info = int(self._keepdims == 4) * 60 * 60
            minute_info = int(self._keepdims == 5) * 60
            second_info = int(self._keepdims == 6)
            time_stamp += (day_info + hour_info + minute_info + second_info) * self._timethrough
            
            return time.localtime(time_stamp)
            
        if self. _keepdims == 2:
            if self._timethrough >11 or self._timethrough < -11:
                raise ValueError('when keep month,timethrough should be -11 to 11')
                
            _month = _time.tm_mon +  self._timethrough
            _year = _time.tm_year
            if _month > 12:
                _year += 1
                _month -= 12
            if _month < 1:
                _year -= 1
                _month += 12
            
            return time.strptime('{}{}'.format(_year,_month),'%Y%m')
            
        
    def fit_transform(self,X):
        if isinstance(self._timeseries_colums,str):
            try:
                self._timeseries_colums = [i for i in range(len(X.columns)) if X.columns[i]==self._timeseries_colums][0]
            except:
                raise ValueError('timeseries_colums is not found in data')
        
        self.type_confirm(X)
        X = np.array(X)

        timescol = X[:,self._timeseries_colums]
        timescol = [time.strptime('{}'.format(i),self._format) for i in timescol]
    
        if self._timethrough != 0:
            timescol = list(map(self.to_timethouth,timescol))

        for dims in range(self._keepdims):
            plus_col = [timestruct[dims] for timestruct in timescol]
            X = np.column_stack((X,plus_col))
            
        if self._plus_weekday == True:
            plus_col = [timestruct.tm_wday + 1 for timestruct in timescol]
            X = np.column_stack((X,plus_col))
        
        if self._plus_yearday == True:
            plus_col = [timestruct.tm_yday + 1 for timestruct in timescol]
            X = np.column_stack((X,plus_col))
            
        if self._x_type == 1:
            self._x_columns.extend(self._struct[:self._keepdims])
            
            if self._plus_weekday == True:
                self._x_columns.append('tm_wday{}'.format(self._timeseries_colums))
            if self._plus_yearday == True:
                self._x_columns.append('tm_yday{}'.format(self._timeseries_colums))
            
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

