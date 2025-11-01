#!/usr/bin/env python3
"""
ATMS 523 Module 5 Project
Radar Parameter Analysis and Rainfall Prediction
Dara Procell
October 28, 2025

Analyzes radar parameters to predict rainfall rates using
multiple machine learning approaches: Linear Regression, Polynomial Regression,
and Random Forest.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import sys
import time


def load_data(filepath):
    """
    Load the radar parameters dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """

    df = pd.read_csv(filepath)
    
    for col in df.columns:
        print(col)
    
    return df

def prepare_features_and_target(df):
    """
    Extract features and target from dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector: R (mm/hr)
    feature_names : list
        List of feature names: Zh (dBZ), Zdr (dB), Ldr (dB), Kdp (deg km-1), Ah (dBZ/km), Adp (dB/km)
    """
   
    feature_cols = []
    target_col = None
    
    for col in df.columns:
        if 'Zh' in col and 'dBZ' in col:
            feature_cols.append(col)
        elif 'Zdr' in col and 'dB' in col and 'dBZ' not in col:
            feature_cols.append(col)
        elif 'Ldr' in col and 'dB' in col:
            feature_cols.append(col)
        elif 'Kdp' in col:
            feature_cols.append(col)
        elif 'Ah' in col and 'dBZ' in col:
            feature_cols.append(col)
        elif ('Adr' in col or 'Adp' in col) and 'dB/km' in col:
            feature_cols.append(col)
        elif 'R' in col and 'mm' in col:
            target_col = col
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Remove Nans
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_cols

def calculate_baseline(Zh, y_true):
    """
    Calculate baseline predictions using Z-R relationship: Z = 200 * R^1.6
    
    Parameters:
    -----------
    Zh : numpy.ndarray
        Radar reflectivity in dBZ
    y_true : numpy.ndarray
        True rainfall rates
        
    Returns:
    --------
    y_pred : numpy.ndarray
        Predicted rainfall rates
    r2 : float
        R-squared score
    rmse : float
        Root mean square error
    """
    # Convert from dBZ to Z
    # dBZ = 10 * log10(Z), so Z = 10^(dBZ/10)
    Z = 10**(Zh/10)
    
    # Apply Z-R relationship: Z = 200 * R^1.6
    # Solving for R: R = (Z/200)^(1/1.6)
    R_pred = (Z/200)**(1/1.6)
    
    r2 = r2_score(y_true, R_pred)
    rmse = np.sqrt(mean_squared_error(y_true, R_pred))
    
    return R_pred, r2, rmse

def split_data(X, y):
    """
    Task 1: Split data into 70-30 train-test split
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Split datasets
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def linear_reg(X_train, y_train, X_test, y_test, feature_names):
    """
    Task 2: Train linear regression and compare with baseline
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and testing features
    y_train, y_test : numpy.ndarray
        Training and testing targets
    feature_names : list
        List of feature names
        
    Returns:
    --------
    results : dict
        Dictionary containing model and performance metrics
    """
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)
  
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("Linear Regression Results:")
    print("Training R²:", train_r2)
    print("Testing R²:", test_r2)
    print("Training RMSE:", train_rmse, " mm/hr")
    print("Testing RMSE:", test_rmse, " mm/hr")
    
    # Calculate baseline (using Zh column)
    # Find index of Zh column
    zh_idx = None
    for i, name in enumerate(feature_names):
        if 'Zh' in name:
            zh_idx = i
            break
    

    Zh_test = X_test[:, zh_idx]
    baseline_pred, baseline_r2, baseline_rmse = calculate_baseline(Zh_test, y_test)
        
    print("Baseline Model Results (Z-R relationship):")
    print("Testing R²:", baseline_r2)
    print("Testing RMSE:", baseline_rmse, "mm/hr")
    
    print("Linear Regression R² improvement:", (test_r2 - baseline_r2))
    print("Linear Regression RMSE improvement:", (baseline_rmse - test_rmse), "mm/hr")

    
    return {
        'model': lr,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'baseline_r2': baseline_r2,
        'baseline_rmse': baseline_rmse
    }

def polynomial_reg(X_train, y_train, X_test, y_test):
    """
    Task 3: Polynomial regression with grid search (orders 0-9)
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and testing features
    y_train, y_test : numpy.ndarray
        Training and testing targets
        
    Returns:
    --------
    results : dict
        Dictionary containing best model and performance metrics
    """

    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    # Parameter grid - skip degree 0 to avoid errors
    param_grid = {'poly__degree': range(1, 10)}
    
    # Grid search
    grid_search = GridSearchCV(
        poly_pipeline,
        param_grid,
        cv=7,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_degree = grid_search.best_params_['poly__degree']
    best_model = grid_search.best_estimator_
    
    print("Best polynomial degree:", best_degree)
    print("Best CV score:", grid_search.best_score_)
    
    y_test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("Best Polynomial Model Performance:")
    print("Testing R²:", test_r2)
    print("Testing RMSE:", test_rmse, "mm/hr")
    
    for i, degree in enumerate(param_grid['poly__degree']):
        mean_score = grid_search.cv_results_['mean_test_score'][i]
        std_score = grid_search.cv_results_['std_test_score'][i]
        print(f"  Degree {degree}: R² = {mean_score:.4f} (+/- {std_score:.4f})")
    
    return {
        'model': best_model,
        'best_degree': best_degree,
        'cv_score': grid_search.best_score_,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'cv_results': grid_search.cv_results_
    }

def random_forest_func(X_train, y_train, X_test, y_test):
    """
    Task 4: Random Forest with grid search
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and testing features
    y_train, y_test : numpy.ndarray
        Training and testing targets
        
    Returns:
    --------
    results : dict
        Dictionary containing best model and performance metrics
    """

    # Define parameter grid
    param_grid = {
        "bootstrap": [True, False],
        "max_depth": [10, 100],
        "max_features": ["sqrt", 1.0],
        "min_samples_leaf": [1, 4],
        "min_samples_split": [2, 10],
        "n_estimators": [200, 1000]
    }
    
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=7,
        scoring='r2',
        n_jobs=-1,
        verbose=2 
    )
    
    grid_search.fit(X_train, y_train)
        
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print("Best CV score:", grid_search.best_score_)
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("Best Random Forest Model Performance:")
    print("Training R²:", train_r2)
    print("Testing R²:", test_r2)
    print("Training RMSE:", train_rmse, "mm/hr")
    print("Testing RMSE:", test_rmse, "mm/hr")
    
    # Feature importance
    print("Feature Importance:")
    for i, importance in enumerate(best_model.feature_importances_):
        print(f"  Feature {i}: {importance:.4f}")
    
    return {
        'model': best_model,
        'best_params': best_params,
        'cv_score': grid_search.best_score_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'feature_importances': best_model.feature_importances_
    }

def create_comparison_plots(results_dict, X_test, y_test):
    """
    Create comparison plots for all models
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from all models
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test targets
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison: Predicted vs Actual Rainfall', fontsize=14, fontweight='bold')
    
    models_to_plot = []
    titles = []
    
    if 'linear' in results_dict:
        models_to_plot.append(results_dict['linear']['model'])
        titles.append(f"Linear Regression\nR² = {results_dict['linear']['test_r2']:.3f}")
    
    if 'polynomial' in results_dict:
        models_to_plot.append(results_dict['polynomial']['model'])
        deg = results_dict['polynomial']['best_degree']
        titles.append(f"Polynomial (degree={deg})\nR² = {results_dict['polynomial']['test_r2']:.3f}")
    
    if 'random_forest' in results_dict:
        models_to_plot.append(results_dict['random_forest']['model'])
        titles.append(f"Random Forest\nR² = {results_dict['random_forest']['test_r2']:.3f}")
    
    if 'linear' in results_dict and results_dict['linear']['baseline_r2'] is not None:
        titles.append(f"Baseline (Z-R)\nR² = {results_dict['linear']['baseline_r2']:.3f}")
    
    for idx, (ax, model, title) in enumerate(zip(axes.flat[:len(models_to_plot)+1], 
                                                  models_to_plot + [None], 
                                                  titles)):
        if model is not None:
            y_pred = model.predict(X_test)
        elif idx == len(models_to_plot): 
            continue
        
        if model is not None:
            ax.scatter(y_test, y_pred, alpha=0.5, s=10)
            
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Rainfall (mm/hr)')
            ax.set_ylabel('Predicted Rainfall (mm/hr)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(models_to_plot)+1, 4):
        axes.flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_final_summary(results_dict):
    """
    Print final summary comparing all models
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from all models
    """

    print("Model Performance Summary:")
    print(f"{'Model':<20} {'Test R²':>10} {'Test RMSE':>12}")

    
    if 'linear' in results_dict:
        if results_dict['linear']['baseline_r2'] is not None:
            print(f"{'Baseline (Z-R)':<20} {results_dict['linear']['baseline_r2']:>10.4f} "
                  f"{results_dict['linear']['baseline_rmse']:>12.4f}")
        
        print(f"{'Linear Regression':<20} {results_dict['linear']['test_r2']:>10.4f} "
              f"{results_dict['linear']['test_rmse']:>12.4f}")
    
    if 'polynomial' in results_dict:
        deg = results_dict['polynomial']['best_degree']
        print(f"{'Polynomial (d=' + str(deg) + ')':<20} "
              f"{results_dict['polynomial']['test_r2']:>10.4f} "
              f"{results_dict['polynomial']['test_rmse']:>12.4f}")
    
    if 'random_forest' in results_dict:
        print(f"{'Random Forest':<20} {results_dict['random_forest']['test_r2']:>10.4f} "
              f"{results_dict['random_forest']['test_rmse']:>12.4f}")
    
    best_r2 = -np.inf
    best_model = None
    
    for model_name in ['linear', 'polynomial', 'random_forest']:
        if model_name in results_dict:
            if results_dict[model_name]['test_r2'] > best_r2:
                best_r2 = results_dict[model_name]['test_r2']
                best_model = model_name
    
    print(f"Best performing model: {best_model.replace('_', ' ').title()}")
    print(f"with R² = {best_r2:.4f}")

def main():
    """
    Main execution function
    """
    
    filepath = 'radar_parameters.csv'

    df = load_data(filepath)

    try:
        X, y, feature_names = prepare_features_and_target(df)
    except Exception as e:
        print(f"\nError preparing data: {e}")
        return
    
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Dictionary to store all results
    results = {}
    
    # Linear Regression
    results['linear'] = linear_reg(X_train, y_train, X_test, y_test, feature_names)
    
    # Polynomial Regression
    results['polynomial'] = polynomial_reg(X_train, y_train, X_test, y_test)

    # Random Forest
    results['random_forest'] = random_forest_func(X_train, y_train, X_test, y_test)
    
    create_comparison_plots(results, X_test, y_test)

    print_final_summary(results)
    
if __name__ == "__main__":
    main()
