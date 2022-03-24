from math import log
from os import link
import statsmodels.api as sm 
import pandas as pd
import numpy as np
from statsmodels.genmod.families.links import inverse_power
import dataset
from contextlib import redirect_stdout
import sklearn 
from sklearn import linear_model 
from statsmodels.api import OLS




def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square( y_true - y_true.mean() ).sum()
    return (1 - (press / sst)) 

def r2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    mean =  y_true.mean()
    sst  = np.square( y_true - y_true.mean()).sum()
    return 1 - sse/sst

x_data, y_data = dataset.get_full_log_xy_data_P()
x_data = np.nan_to_num(x_data)
y_data = np.nan_to_num(y_data)
x_data, y_data = dataset.replace_missing(x_data, y_data)
x_data, y_data = dataset.normalize(x_data, y_data)
x_data = sm.add_constant(x_data)



'''#Gaussian family eith inverse_power link 
gaussian_model = sm.GLM(y_data, x_data, family=sm.families.Gaussian(link=inverse_power()))
result = gaussian_model.fit()
predicted_values = result.predict(x_data)
print(predicted_values)
print(y_data.T)
A = np.identity(len(result.params))
A = A[1:,:]'''


linear = linear_model.LinearRegression()
linear.fit(x_data, y_data)
predicted_values = linear.predict(x_data)
print(predicted_values.T)
print(y_data.T)


with open(r'C:\\Users\\isjom\\Desktop\\YOYOYO\\Results_GLM\\Results_GLM_Family=Gaussian_log_inverse_power.txt', 'w') as f:
        with redirect_stdout(f):
            print(OLS(y_data,x_data).fit().summary())
            print(r2(y_data.T, predicted_values))
            print(predicted_r2(y_data.T, predicted_values, x_data))
            #print(linear.f_test(A))
