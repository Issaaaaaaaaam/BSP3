import pandas as pd
import numpy as np 
import sklearn 
from sklearn import linear_model 
import dataset
from sklearn.metrics import r2_score
import pickle 
from statsmodels.api import OLS
from contextlib import redirect_stdout
import statsmodels.api as sm 


def linear_regression(name_of_save_file,name_of_statistic_file, x): 
    
    x_data, y_data = dataset.get_full_xy_data()

    x_data = np.nan_to_num(x_data)
    y_data = np.nan_to_num(y_data)

    x_data, y_data = dataset.replace_missing(x_data, y_data)
    x_data, y_data = dataset.normalize(x_data, y_data)
    best = 0
    for _ in range (x): 
        training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
        
        linear = linear_model.LinearRegression()
        linear.fit(x_data, y_data)
        acc = linear.score(x_data, y_data)
        print(acc)

        if acc > best:
            best = acc
            with open(name_of_save_file, "wb") as f: 
                pickle.dump(linear, f)

    pickle_in = open(name_of_save_file, "rb")
    linear = pickle.load(pickle_in)

    with open(name_of_statistic_file, 'w') as f:
        with redirect_stdout(f):
            print(OLS(y_data,x_data).fit().summary())

def linear_regression_P(name_of_save_file,name_of_statistic_file, x): 
    
    x_data, y_data = dataset.get_full_xy_data_P()

    x_data = np.nan_to_num(x_data)
    y_data = np.nan_to_num(y_data)

    x_data, y_data = dataset.replace_missing(x_data, y_data)
    x_data, y_data = dataset.normalize(x_data, y_data)
    x_data = sm.add_constant(x_data)
    best = 0
    for _ in range (x): 
        training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
        
        linear = linear_model.LinearRegression()
        linear.fit(x_data, y_data)
        acc = linear.score(x_data, y_data)
        print(acc)

        if acc > best:
            best = acc
            with open(name_of_save_file, "wb") as f: 
                pickle.dump(linear, f)

    pickle_in = open(name_of_save_file, "rb")
    linear = pickle.load(pickle_in)

    with open(name_of_statistic_file, 'w') as f:
        with redirect_stdout(f):
            print(OLS(y_data,x_data).fit().summary())

def linear_regression_log(name_of_save_file,name_of_statistic_file, x): 
    
    x_data, y_data = dataset.get_full_log_xy_data()

    x_data = np.nan_to_num(x_data)
    y_data = np.nan_to_num(y_data)

    x_data, y_data = dataset.replace_missing(x_data, y_data)
    x_data, y_data = dataset.normalize(x_data, y_data)

    best = 0
    for _ in range (x): 
        training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
        
        linear = linear_model.LinearRegression()
        linear.fit(x_data, y_data)
        acc = linear.score(x_data, y_data)
        print(acc)

        if acc > best:
            best = acc
            with open(name_of_save_file, "wb") as f: 
                pickle.dump(linear, f)
                
    pickle_in = open(name_of_save_file, "rb")
    linear = pickle.load(pickle_in)
    with open( name_of_statistic_file, 'w') as f:
        with redirect_stdout(f):
            print(OLS(y_data,x_data).fit().summary())

def linear_regression_logs_P(name_of_save_file,name_of_statistic_file, x): 
    
    x_data, y_data = dataset.get_full_log_xy_data_P()

    x_data = np.nan_to_num(x_data)
    y_data = np.nan_to_num(y_data)

    x_data, y_data = dataset.replace_missing(x_data, y_data)
    x_data, y_data = dataset.normalize(x_data, y_data)
    x_data = sm.add_constant(x_data)
    best = 0
    for _ in range (x): 
        training_x, training_y, validation_x, validation_y = dataset.get_training_and_test_set(x_data, y_data)
        
        linear = linear_model.LinearRegression()
        linear.fit(x_data, y_data)
        acc = linear.score(x_data, y_data)
        print(acc)

        if acc > best:
            best = acc
            with open(name_of_save_file, "wb") as f: 
                pickle.dump(linear, f)
                
    pickle_in = open(name_of_save_file, "rb")
    linear = pickle.load(pickle_in)

    with open(name_of_statistic_file, 'w') as f:
        with redirect_stdout(f):
            print(OLS(y_data,x_data).fit().summary())




linear_regression_P("unified_result_log_values_P.pickle","unified_result_log_values_P.txt",2)
