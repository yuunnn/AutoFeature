import numpy as np

def fillna(x):
    def foo(y):
        try:
            if np.isnan(y):
                return -999
        except:
            pass
        return y
    return np.vectorize(lambda v: foo(v))(x)