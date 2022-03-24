from math import log
from os import link
import statsmodels.api as sm 
import pandas as pd
import numpy as np
from statsmodels.genmod.families.links import inverse_power
import dataset
from contextlib import redirect_stdout

x_data, y_data = dataset.get_full_log_xy_data()
x_data = np.nan_to_num(x_data)
y_data = np.nan_to_num(y_data)
x_data, y_data = dataset.replace_missing(x_data, y_data)
x_data, y_data = dataset.normalize(x_data, y_data)
X = sm.add_constant(x_data)
gaussian_model = sm.GLM(y_data, X, family=sm.families.Gaussian(link=inverse_power()))
gaussian_results = gaussian_model.fit()
A = np.identity(len(gaussian_results.params))
A = A[1:,:]





with open(r'C:\\Users\\Issam\\Desktop\\VsCode\\Python_code\\Results_GLM\\Results_GLM_Family=Gaussian_log_inverse_power_DEMO.txt', 'w') as f:
        with redirect_stdout(f):
            print(gaussian_results.summary())
            print(gaussian_results.summary2())
            print(gaussian_results.f_test(A))
