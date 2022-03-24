import statsmodels.api as sm 
import pandas as pd
import numpy as np
from statsmodels.genmod.families.links import inverse_power
import dataset
from contextlib import redirect_stdout
import sklearn 
from sklearn import linear_model 
from statsmodels.api import OLS
from sklearn.metrics import r2_score


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

x_data, y_data = dataset.get_full_log_xy_data_2_3_4()
x_data = np.nan_to_num(x_data)
y_data = np.nan_to_num(y_data)
x_data, y_data = dataset.replace_missing(x_data, y_data)
x_data, y_data = dataset.normalize(x_data, y_data)
x_data = sm.add_constant(x_data)




results = []

i = 0 
''' Linear regression 
for i in range (29):
    training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
    linear = linear_model.LinearRegression()
    linear.fit(training_x, training_y)
    predicted_values = linear.predict(validation_x)
    print(r2_score(validation_y,predicted_values))
    result.append(r2_score(validation_y,predicted_values))
    print(result)
    i+=1
'''
# GLM
for i in range (29):
    training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
    gaussian_model = sm.GLM(training_y, training_x, family=sm.families.Gaussian(link=inverse_power()))
    result = gaussian_model.fit()
    predicted_values = result.predict(validation_x)
    validation_y = np.squeeze(validation_y) 
    print(r2_score(validation_y.T,predicted_values))
    results.append(r2_score(validation_y.T,predicted_values))
    print(results)
    i+=1

#By commenting out one for loop or the other we obtain GLM results or linear regression

results = np.array(results)
print("------------------------------------------------------------------------\n")
print("-----------------------Starting of the results--------------------------")
print("min: ", np.min(results), "\n")
print("max: ", np.max(results), "\n")
print("mean: ", np.mean(results), "\n")
print("std: ", np.std(results), "\n")
