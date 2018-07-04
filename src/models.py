
'''
Model functions
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import optunity
import optunity.metrics

# define constants
SEED = 15

def py_rf_default(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def py_rf_rParams(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, n_estimators=500, max_features=1/3)
    forest_reg.fit(x_train, y_train)
    # validate
    y_pred = forest_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def rf_randomsearch(x_train, y_train, x_valid):
    # Random forest
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    param_grid = {'n_estimators': [3, 100, 300, 500], 'max_features': 0.2, 0.4, 0.6, 0.8, 1]}
    reg = RandomizedSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_absolute_error')
    reg.fit(x_train, y_train)
    # validate
    y_pred = reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def rf_pso(x_train, y_train, x_valid):
    # Random forest using particle swarm opt on the hyperparameters
    # pso
    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def performance(x_train, y_train, x_test, y_test,n_estimators, max_features):
        model = RandomForestRegressor(n_estimators=int(n_estimators),
                                       max_features=max_features)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return(optunity.metrics.mse(y_test, predictions))
    # hyperparameter search
    optimal_pars, _, _ = optunity.minimize(performance, num_evals = 100, n_estimators=[10,800], max_features=[0.2,10])
    # train
    forest_reg = RandomForestRegressor(random_state=SEED, **optimal_pars)
    reg.fit(x_train, y_train)
    # validate
    y_pred = reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def rf_gridsearch(x_train, y_train, x_valid):
    # Random forest using grid search
    # train
    forest_reg = RandomForestRegressor(random_state=SEED)
    param_grid = {'n_estimators': [3, 250, 500], 'max_features': [0.2, 0.4, 0.6, 0.8, 1]}
    forest_grid = GridSearchCV(forest_reg, param_grid, cv=5, scoring='mean_absolute_error')
    forest_grid.fit(x_train, y_train)
    # validate
    y_pred = forest_grid.predict(x_valid)
    # return predictive probabilities
    return(y_pred)

def gbm_rf_default(x_train, y_train, x_valid):
    # Gradient Boosting Random Forest
    # train
    gbm_reg = GradientBoostingRegressor(random_state=SEED)
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)


def xgboost_rf(x_train, y_train, x_valid):
    # Random forest
    from xgboost import XGBRegressor
    # train
    gbm_reg = XGBRegressor(random_state=SEED)
    gbm_reg.fit(x_train, y_train)
    # validate
    y_pred = gbm_reg.predict(x_valid)
    # return predictive probabilities
    return(y_pred)
