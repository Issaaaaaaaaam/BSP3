import pandas as pd
import numpy as np
from math import log
import random
import copy


# module for extracting the x and y data of the dataset
# as well as providing training and validation sets

# in this code x_index is considered an row of the table
# and column_index is the index of the different column in the table


def get_full_xy_data():
    df = pd.read_csv('bank_data_final.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan'], row['net_loan_1'], row['net_loan_2'], row['net_loan_3'],
            row['loss_allow'], row['loss_allow_1'], row['loss_allow_2'], row['loss_allow_3'],
            row['dep'], row['dep_1'], row['dep_2'], row['dep_3'],
            row['yield_ea'], row['yield_ea_1'], row['yield_ea_2'], row['yield_ea_3'],
            row['fundc_ea'], row['fundc_ea_1'], row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa'], row['inc_aa_1'], row['inc_aa_2'], row['inc_aa_3'],
            row['CAR'], row['CAR_1'], row['CAR_2'], row['CAR_3'],
            row['tot_asst'], row['tot_asst_1'], row['tot_asst_2'], row['tot_asst_3'],
            row['tot_eq'], row['tot_eq_1'], row['tot_eq_2'], row['tot_eq_3'],
            row['tot_loan'], row['tot_loan_1'], row['tot_loan_2'], row['tot_loan_3'],
            row['risk_dens'], row['risk_dens_1'], row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth'], row['GDP_growth_1'], row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth'], row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP'], row['debt_GDP_1'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP'], row['govex_GDP_1'], row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat'], row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth'], row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp'], row['unemp_1'], row['unemp_2'], row['unemp_3'],
            row['yield_10Y'], row['yield_10Y_1'], row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret'], row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def get_full_xy_data_P():
    df = pd.read_csv('bank_data_final.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan'], 
            row['loss_allow'], row['loss_allow_1'], row['loss_allow_3'],
            row['dep'], row['dep_1'], 
            row['yield_ea'], row['yield_ea_1'], row['yield_ea_2'], 
            row['fundc_ea'], row['fundc_ea_3'],
            row['inc_aa'], row['inc_aa_2'],
            row['CAR'], row['CAR_1'], 
            row['tot_asst_2'], row['tot_asst_3'],
            row['tot_loan_2'], row['tot_loan_3'],
            row['risk_dens'],
            row['GDP_growth'], row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth'], row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP_1'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP_1'], row['govex_GDP_2'],
            row['inflat'], row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth'], row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp'], row['unemp_1'], row['unemp_2'], row['unemp_3'],
            row['yield_10Y'], row['yield_10Y_1'], row['yield_10Y_3'],
            row['S&P500_ret'], row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data   

def get_full_log_xy_data():
    df = pd.read_csv('bank_data_final_log.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan'], row['net_loan_1'], row['net_loan_2'], row['net_loan_3'],
            row['loss_allow'], row['loss_allow_1'], row['loss_allow_2'], row['loss_allow_3'],
            row['dep'], row['dep_1'], row['dep_2'], row['dep_3'],
            row['yield_ea'], row['yield_ea_1'], row['yield_ea_2'], row['yield_ea_3'],
            row['fundc_ea'], row['fundc_ea_1'], row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa'], row['inc_aa_1'], row['inc_aa_2'], row['inc_aa_3'],
            row['CAR'], row['CAR_1'], row['CAR_2'], row['CAR_3'],
            row['tot_asst'], row['tot_asst_1'], row['tot_asst_2'], row['tot_asst_3'],
            row['tot_eq'], row['tot_eq_1'], row['tot_eq_2'], row['tot_eq_3'],
            row['tot_loan'], row['tot_loan_1'], row['tot_loan_2'], row['tot_loan_3'],
            row['risk_dens'], row['risk_dens_1'], row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth'], row['GDP_growth_1'], row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth'], row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP'], row['debt_GDP_1'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP'], row['govex_GDP_1'], row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat'], row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth'], row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp'], row['unemp_1'], row['unemp_2'], row['unemp_3'],
            row['yield_10Y'], row['yield_10Y_1'], row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret'], row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data



def get_full_log_xy_data_P():
    df = pd.read_csv('bank_data_final_log.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan'], row['net_loan_1'], row['net_loan_3'],
            row['loss_allow'], row['loss_allow_1'],
            row['dep_3'],
            row['yield_ea_1'], row['yield_ea_2'],
            row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa'], row['inc_aa_1'], row['inc_aa_2'],
            row['CAR'], row['CAR_1'], row['CAR_2'], row['CAR_3'],
            row['tot_asst'], row['tot_asst_1'], row['tot_asst_2'], row['tot_asst_3'],
            row['tot_eq'], row['tot_eq_2'],
            row['tot_loan'], row['tot_loan_3'],
            row['risk_dens'], row['risk_dens_1'], row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth'], row['GDP_growth_1'], row['GDP_growth_3'],
            row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat'], row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth'], row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp'], row['unemp_1'], row['unemp_3'],
            row['yield_10Y'], row['yield_10Y_1'], row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret'], row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def get_reduced_xy_data():
    # below code is redundant since we have saved the file as csv
    # my_sheet = 'all_years'
    # file_name = 'bank_data_final.xlsx'
    # df = pd.read_excel(file_name, sheet_name=my_sheet)
    # df.to_csv('bank_data_final.csv')

    # read from csv is way faster
    df = pd.read_csv('bank_data_final.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['yield_ea'], row['yield_ea_1'], row['yield_ea_2'], row['yield_ea_3'],
            row['fundc_ea'], row['fundc_ea_1'], row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa'], row['inc_aa_1'], row['inc_aa_2'], row['inc_aa_3'],
            row['CAR'], row['CAR_1'], row['CAR_2'], row['CAR_3'],
            row['risk_dens'], row['risk_dens_1'], row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth'], row['GDP_growth_1'], row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth'], row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP'], row['debt_GDP_1'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP'], row['govex_GDP_1'], row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat'], row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth'], row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp'], row['unemp_1'], row['unemp_2'], row['unemp_3'],
            row['yield_10Y'], row['yield_10Y_1'], row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret'], row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def get_full_xy_data_4():
    df = pd.read_csv('bank_data_final_log.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan_3'],
            row['loss_allow_3'],
            row['dep_3'],
            row['yield_ea_3'],
            row['fundc_ea_3'],
            row['inc_aa_3'],
            row['CAR_3'],
            row['tot_asst_3'],
            row['tot_eq_3'],
            row['tot_loan_3'],
            row['risk_dens_3'],
            row['GDP_growth_3'],
            row['export_growth_3'],
            row['debt_GDP_3'],
            row['govex_GDP_3'],
            row['inflat_3'],
            row['HPI_growth_3'],
            row['unemp_3'],
            row['yield_10Y_3'],
            row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def get_full_log_xy_data_4_3():
    df = pd.read_csv('bank_data_final_log.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan_2'], row['net_loan_3'],
            row['loss_allow_2'], row['loss_allow_3'],
            row['dep_2'], row['dep_3'],
            row['yield_ea_2'], row['yield_ea_3'],
            row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa_2'], row['inc_aa_3'],
            row['CAR_2'], row['CAR_3'],
            row['tot_asst_2'], row['tot_asst_3'],
            row['tot_eq_2'], row['tot_eq_3'],
            row['tot_loan_2'], row['tot_loan_3'],
            row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat_2'], row['inflat_3'],
            row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp_2'], row['unemp_3'],
            row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def get_full_log_xy_data_2_3_4():
    df = pd.read_csv('bank_data_final_log.csv')

    # initialise variables
    x_data = []
    y_data = []

    for index, row in df.iterrows():
        x = [
            row['net_loan_1'], row['net_loan_2'], row['net_loan_3'],
            row['loss_allow_1'], row['loss_allow_2'], row['loss_allow_3'],
            row['dep_1'], row['dep_2'], row['dep_3'],
            row['yield_ea_1'], row['yield_ea_2'], row['yield_ea_3'],
            row['fundc_ea_1'], row['fundc_ea_2'], row['fundc_ea_3'],
            row['inc_aa_1'], row['inc_aa_2'], row['inc_aa_3'],
            row['CAR_1'], row['CAR_2'], row['CAR_3'],
            row['tot_asst_1'], row['tot_asst_2'], row['tot_asst_3'],
            row['tot_eq_1'], row['tot_eq_2'], row['tot_eq_3'],
            row['tot_loan_1'], row['tot_loan_2'], row['tot_loan_3'],
            row['risk_dens_1'], row['risk_dens_2'], row['risk_dens_3'],
            row['GDP_growth_1'], row['GDP_growth_2'], row['GDP_growth_3'],
            row['export_growth_1'], row['export_growth_2'], row['export_growth_3'],
            row['debt_GDP_1'], row['debt_GDP_2'], row['debt_GDP_3'],
            row['govex_GDP_1'], row['govex_GDP_2'], row['govex_GDP_3'],
            row['inflat_1'], row['inflat_2'], row['inflat_3'],
            row['HPI_growth_1'], row['HPI_growth_2'], row['HPI_growth_3'],
            row['unemp_1'], row['unemp_2'], row['unemp_3'],
            row['yield_10Y_1'], row['yield_10Y_2'], row['yield_10Y_3'],
            row['S&P500_ret_1'], row['S&P500_ret_2'], row['S&P500_ret_3']
        ]
        y = row['CAR_T1']

        # doesn't accept values that contain a null value or outliers
        x_data.append(x)
        y_data.append([y])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


# remove data with less than 30% data
# outliers = False return x_data and y_data without outliers
# outliers = True  return x_data and y_data and x_outliers and y_outliers
def remove_outliers(x_data, y_data, outliers=False):
    x_outlier = []
    y_outlier = []

    # size = number of columns/values a single row/x_value has + 1 because of y_data
    size = len(x_data[0]) + 1

    # remove values with less than 30% data
    for x_index in range(len(x_data)):

        # count amount of values that aren't missing
        # and remove the row if less than 30% of the values are present
        count_missing = 0
        for column_index in range(len(x_data[x_index])):
            if np.isnan(x_data[x_index][column_index]):
                count_missing += 1
        if count_missing > size / 100 * 30:
            x_outlier.append(x_data[x_index])
            y_outlier.append(y_data[x_index][0])
            print("x_index " + str(x_index))
            x_data = np.delete(x_data, x_index, axis=0)
            y_data = np.delete(y_data, x_index)

    if outliers:
        return x_data, y_data, x_outlier, y_outlier
    else:
        return x_data, y_data


# partitioned_X returns a list containing all of the values of x_data partitioned
# into different lists according to the different columns
def partition_X(x_data):
    # if x_data is empty raise an error
    if x_data is None or x_data is not x_data:
        raise Exception("x_data is empty")

    # all_values is initialised with len(x) amount of empty lists
    # to those empty lists the values of x are then added
    all_values = []
    for value_index in range(len(x_data[0])):
        all_values.append([])

    # partition all the different values
    for x_index in range(len(x_data)):
        for column_index in range(len(x_data[0])):
            all_values[column_index].append(x_data[x_index][column_index])

    return all_values


def replace_missing(x_data, y_data):
    # calculate all the means/maximums of the different columns and put them into a list
    means = []
    maxs = []

    for values in partition_X(x_data):
        # nanmean ignores NaN values
        means.append(np.nanmean(values))
        # nanmax ignores NaN values
        maxs.append(np.nanmax(values))

    for x_index in range(len(x_data)):
        # column_index is the index of the corresponding column
        for column_index in range(len(x_data[x_index])):
            # since pandas cannot differentiate between zero division and missing values
            # because both get replaced by NaN, the mean is used to replace both of them
            # instead of replacing missing values with the max and incorrect values with the mean
            if np.isnan(x_data[x_index][column_index]):
                x_data[x_index][column_index] = means[column_index]
            # below code can't be used because of above mentioned reasons
            # if x_data[x_index][column_index] = "MISSING":
            #    x_data[x_index][column_index] = maxs[column_index]
        # check for y value, len(x_data) can be used since len(x_data) and len(y_data) have the same size
        if np.isnan(y_data[x_index][0]):
            y_data[x_index] = [np.mean(y_data)]

    return x_data, y_data


def calculateLogMinMaxT(x, minimum, maximum):
    return log(abs(min(0, minimum)) + x + 1, abs(min(0, minimum)) + maximum + 1)


def normalize(x_data, y_data):
    # calculate all the maximums/minimums of the different columns and put them into a list
    maxs = []
    mins = []
    for values in partition_X(x_data):
        # nanmean ignores NaN values
        mins.append(np.nanmin(values))
        # nanmax ignores NaN values
        maxs.append(np.nanmax(values))

    # normalize x_data
    for x_index in range(len(x_data)):
        for column_index in range(len(x_data[x_index])):
            x_data[x_index][column_index] = calculateLogMinMaxT(x_data[x_index][column_index], mins[column_index],
                                                                maxs[column_index])

    # normalize y_data
    for y_index in range(len(y_data)):
        y_data[y_index] = calculateLogMinMaxT(y_data[y_index], np.nanmin(y_data), np.nanmax(y_data))

    return x_data, y_data


def get_full_names():
    return ["net_loan", "loss_allow", "dep", "yield_ea", "fundc_ea", "inc_aa", "CAR", "tot_asst", "tot_eq",
            "tot_loan", "risk_dens", "GDP_growth", "export_growth", "debt_GDP", "govex_GDP", "inflat",
            "HPI_growth", "unemp", "yield_10Y", "SandP500_ret"]


def get_reduced_names():
    return ["yield_ea", "fundc_ea", "inc_aa", "CAR","risk_dens",
            "GDP_growth", "export_growth", "debt_GDP", "govex_GDP",
            "inflat", "HPI_growth", "unemp", "yield_10Y", "SandP500_ret"]


def adhoc_X(x_data):
    # assumes we have the same variable at 4 different points in time
    # all_values is initialised with len(x)//4 amount of empty lists
    # to those empty lists the values of x are then added
    all_values = []

    print(len(x_data))

    for column_index in range(len(x_data[0]) // 4):
        all_values.append([])
        for x_index in range(0, len(x_data)):
            # at all_values[x_index] append an empty list
            all_values[column_index].append(x_data[x_index][column_index * 4:column_index * 4 + 4])
        # column values that are divided by categories get added to all_values[x_index]

    # make an numpy array out of all_values
    all_values = np.array(all_values)

    # create a dictionary
    result = {}

    # get the names using get_reduced_names() or get_full_names()
    names = get_reduced_names()

    for name_index in range(len(names)):
        # the keys to the dictionary are "in_" + the name of the category because that's the name of the input neurons
        # the values of the dictionary are the arrays containing all the x_values of that category
        # in the form of (amount of x values,4)
        result["in_" + names[name_index]] = all_values[name_index]

    return result


def getIndexNames():
    names = get_full_names()
    result = []
    for name in names:
        for i in range(1, 5):
            result.append(name + "_" + str(i))

    return result


def get_training_and_test_set(x_data, y_data):
    # can not accept x_adhoc datasets

    shuffle_set = []

    for i in range(x_data.shape[0]):
        shuffle_set.append(i)

    random.shuffle(shuffle_set)
    # randomly shuffle the x_data as well as the y_data
    # where every x and y pair keep the same values only the position changes
    for x_index in range(len(shuffle_set)):
        # needs a deepcopy of x or else the result will be 2 times x_data[x_index]
        copy_of_x = copy.deepcopy(x_data[x_index])
        x_data[x_index], x_data[shuffle_set[x_index]] = x_data[shuffle_set[x_index]], copy_of_x

        # same thing with y
        copy_of_y = copy.deepcopy(y_data[x_index])
        y_data[x_index], y_data[shuffle_set[x_index]] = y_data[shuffle_set[x_index]], copy_of_y

    # initialise values
    training_set_x = []
    training_set_y = []

    # put the first 70% into the training_set
    for x_index in range((x_data.shape[0] * 70) // 100):
        training_set_x.append(x_data[x_index])
        training_set_y.append(y_data[x_index])

    # initialise values
    validation_set_x = []
    validation_set_y = []

    # put the rest into the validation set
    for x_index in range((x_data.shape[0] * 70) // 100, x_data.shape[0]):
        validation_set_x.append(x_data[x_index])
        validation_set_y.append(y_data[x_index])

    return np.array(training_set_x), np.array(training_set_y), np.array(validation_set_x), np.array(validation_set_y)
