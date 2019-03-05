import numpy as np
import pandas as pd
from abc import ABC,abstractmethod


class Propecssing(ABC):

    @abstractmethod
    def fit(self):
        pass


class Timeseriesor(Propecssing):
    
    def fit(self,x):
        pass
        

class WindowSlider(Propecssing):

    def fit(self):
        pass

class Labalencoder(Propecssing):
    
    def fit(self):
        pass

class Combinator(Propecssing):

    def fit(self):
        pass

