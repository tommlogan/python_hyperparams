'''
Fit different models to the dataset
'''
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import code
import optunity
import optunity.metrics
from models import *

# define constants
DATA_PATH = 'data/data_zeroinflate.csv'
TIME_PATH = 'data/time_elapsed.csv'
HOLDOUT_NUM = 10
SEED = 15
CORES_NUM = 10 #min(25,int(os.cpu_count()))
PAR = True
RESPONSE_VAR = 'y'

def main():
    # loop different models

    # import the data
    data = import_data()

    # create the holdout datasets
    create_holdouts(data)

    # models to test
    models = [rf_pso, py_rf_default, py_rf_Rparams, rf_randomsearch, rf_gridsearch]
    for model in models:
        # cross validation
        elapsed = cross_validation(data, model)
        print("time to run {} holdouts in python: {} min".format(HOLDOUT_NUM, elapsed/60))

        # write time elapsed
        file_exists = os.path.isfile(TIME_PATH)
        header = ['Model','Time','data']
        with open(TIME_PATH, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([models[0].__name__, elapsed, DATA_PATH])



def import_data():
    # import the data and separate the test and train
    # import data
    data = pd.read_csv(DATA_PATH)

    # convert categorical variables
    var_cat = ['x48', 'x49','x50','x51','x52','x53','x54']
    for v in var_cat:
        data[v] = pd.factorize(data[v])[0]

    return(data)


def create_holdouts(data):
    '''
        creates csvs for the test and train datasets
    '''
    # divide the data
    skfolds = ShuffleSplit(n_splits=HOLDOUT_NUM, random_state=SEED, test_size=0.2)
    # list of indices
    fold_indices = list(skfolds.split(data, data[RESPONSE_VAR]))

    # make a boolean list where it is one if it's in the training set, column number is the holdout Number
    row_nums = pd.DataFrame(dict.fromkeys(range(HOLDOUT_NUM),[False] * len(data)))
    for i in range(HOLDOUT_NUM):
        row_nums.loc[fold_indices[i][0], i] = True

    # save the pd dataframe as a csv so that it can be imported and have models trained on it
    row_nums.to_csv('data/holdout_indices.csv')


def cross_validation(train, model):
    '''
        divide the data into the training and validating sets
        train and test the models specified in the models list
        input: data frame, list of model functions, random_state variable
        return predictive accuracy
    '''
    # import the data divisions
    train_indices = pd.read_csv('data/holdout_indices.csv')
    # create this list of indices with a dummy variable to act as the iterator for saving results
    fold_indices_iterated = []
    for i in range(HOLDOUT_NUM):
        fold_indices_iterated += [(np.where(train_indices[str(i)])[0], np.where(train_indices[str(i)]==False)[0], i)]
    # parallelize the cross-validation
    t = time.time()
    if PAR:
        Parallel(n_jobs=CORES_NUM)(delayed(model_cross_validate)(train, model, train_index, test_index, cv_num) for train_index, test_index, cv_num in fold_indices_iterated)
    else:
        for train_index, test_index, cv_num in fold_indices_iterated:
            model_cross_validate(train, model, train_index, test_index, cv_num)

    return(time.time() - t)


def model_cross_validate(train, model, train_index, test_index, cv_num):
    # divide the data, then train on multiple models
    # split data
    x_train, y_train, x_valid, y_valid = k_fold_data(train, train_index, test_index)

    # init evaluation vectors
    predictions = y_valid.as_matrix()
    predictions = np.expand_dims(predictions, axis=1)
    predictions_names = ['y_valid']

    # fit models
    # model name
    model_name = model.__name__
    # train model and return predictions
    results = model(x_train, y_train, x_valid)
    if len(results.shape) == 1:
        results = np.reshape(results, (len(results),1))
    # append predictive probabilities to np.array
    predictions = np.append(predictions, results, axis=1)
    # append column headings
    predictions_names += [model_name]

    # save result to csv
    results_table = pd.DataFrame(data=predictions, columns=predictions_names)
    results_table.to_csv('data/predictions/{}_py_{}.csv'.format(model_name,cv_num))


def k_fold_data(train, train_index, test_index):
    # return train and validation sets
    x_train = train.iloc[train_index]
    y_train = train[RESPONSE_VAR].iloc[train_index]
    x_valid = train.iloc[test_index]
    y_valid = train[RESPONSE_VAR].iloc[test_index]
    #
    # drop y from x
    x_train = x_train.drop(RESPONSE_VAR, axis=1)
    x_valid = x_valid.drop(RESPONSE_VAR, axis=1)
    #
    return(x_train, y_train, x_valid, y_valid)


if __name__ == '__main__':
    main()
