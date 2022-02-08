import numpy as np


def MAPE(y_test, y_pred):
    mape=np.mean(np.abs((y_test - y_pred) / y_test)) * 100 
    return mape