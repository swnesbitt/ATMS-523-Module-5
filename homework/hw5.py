#!/bin/python3
"""Homework 5 -- Scott Thomas Andersen -- ATMS 523."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def load_data(filename="radar_parameters.csv"):
    """Read the data file."""
    return pd.read_csv(filename, index_col=[0])


def prep_data(dataframe):
    """Prep the data and return X and Y  sets."""
    x = dataframe[['Zh (dBZ)', 'Zdr (dB)', 'Ldr (dB)', 'Kdp (deg km-1)',
                  'Ah (dBZ/km)', 'Adr (dB/km)']].to_numpy()
    y = dataframe['R (mm/hr)'].to_numpy()

    # standardize the data?

    return x, y


def split_data(x, y, seed=251101):
    """Create the train test split of the data."""
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=seed)

    return x_train, x_test, y_train, y_test


def baseline_predict(x):
    """Classify data using baseline classifier."""
    # Zh - radar reflectivity factor (dBZ) - use formula dBZ = 10log_10(Z)
    # Z = 200R^1.6

    # That is,
    # (dBZ / 10)^10 = Z
    # (Z / 200) ^ (1/1.6) = R

    Z = 10**(x[:, 0]/10)
    R = (Z/200)**(1/1.6)
    return R


def multiregression_predict(x_train, y_train, x_test):
    """Predict the values using multi linear regression."""
    # documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    reg = LinearRegression().fit(x_train, y_train)
    return reg.predict(x_test)


def polynomial_predict(x_train, y_train, x_test):
    """Predict by fitting polynomial, use grid search to determine degree."""
    params = {'polynomialfeatures__degree': np.arange(9)}

    # create the model pipeline and prep the grid search
    poly_model = make_pipeline(
            PolynomialFeatures(7),
            LinearRegression()
    )
    grid = GridSearchCV(poly_model, params, cv=7, n_jobs=-1)

    # fit the model and get the best estimator
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    print("Best params: ", grid.best_params_)

    return model.predict(x_test)


def rforest_predict(x_train, y_train, x_test):
    """Predict with random forest regression."""
    param_grid = {
         "bootstrap": [True, False],
         "max_depth": [10, 100],
         "max_features": ["sqrt", 1.0],
         "min_samples_leaf": [1, 4],
         "min_samples_split": [2, 10],
         "n_estimators": [200, 1000]
     }

    # create the model pipeline and prep the grid search
    forest_model = RandomForestRegressor()
    grid = GridSearchCV(forest_model, param_grid, cv=7, n_jobs=-1)

    # fit the model and get the best estimator
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    return model.predict(x_test)


def evaluate(y_pred, y_true, print_scores=True):
    """Calculate the evaluation scores."""
    scores = {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred)
    }

    if print_scores:
        print(f"R2:  {scores['R2']:.4f}")
        print(f"MSE: {scores['MSE']:.4f}")

    return scores


def main():
    """Run main function."""
    df = load_data()
    x, y = prep_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # run the prediction models
    print("Baseline prediction")
    y_pred_base = baseline_predict(x_test)
    evaluate(y_pred_base, y_test)
    print("")
    # print(y_pred_base)
    # print(y_test)
    print("Multi regression prediction")
    y_pred_linr = multiregression_predict(x_train, y_train, x_test)
    evaluate(y_pred_linr, y_test)
    print("")

    # polynomial regression
    print("Polynomial regression")
    y_pred_poly = polynomial_predict(x_train, y_train, x_test)
    evaluate(y_pred_poly, y_test)
    print("")

    # random forest regression
    print("Random forest regression")
    y_pred_poly = rforest_predict(x_train, y_train, x_test)
    evaluate(y_pred_poly, y_test)


if __name__ == "__main__":
    main()
