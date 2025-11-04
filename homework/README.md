# Module 5 Homework Submission by Scott Andersen 
## ATMS 523 -- 2025

This assignment I did not use a jupyter notebook but created a basic python script to complete the exercises. Assuming that scikit learn is installed on your machine you should be able to run as is using ./hw5.py.


## Exercises and discussion
1. Split the data into a 70-30 split for training and testing data.

This was completed using scikit learn's train_test_split function. In hw5.py this is done in the function split_data.

2. Using the split created in (1), train a multiple linear regression dataset using the training dataset, and validate it using the testing dataset.  Compare the $R^2$ and root mean square errors of model on the training and testing sets to a baseline prediction of rain rate using the formula $Z = 200 R^{1.6}$.

The baseline_predict function runs the baseline prediction model and multiregression_predict runs the Multiregression model. 

The results of these two are:
Baseline prediction
R2:  0.3147
MSE: 52.5047

Multi regression prediction
R2:  0.9883
MSE: 0.8977

As we can see the multi regression vastly outperforms the the baseline model, acheiving a higher R2 (closer to one) and a much lower MSE.

3. Repeat 1 doing a grid search over polynomial orders, using a grid search over orders 0-9, and use cross-validation of 7 folds.  For the best polynomial model in terms of $R^2$, does it outperform the baseline and the linear regression model in terms of $R^2$ and root mean square error?

This was implemented with the function polynomial_predict, the best model had a degree of 3, and had the following metrics:

R2:  1.0000
MSE: 0.0002

This is a very well performing model that outperforms the other two quite well.

4. Repeat 1 with a Random Forest Regressor, and perform a grid_search on the following parameters:
   
   ```python
   param_grid = {
    "bootstrap": [True, False],
    "max_depth": [10, 100],
    "max_features": ["sqrt", 1.0],  
    "min_samples_leaf": [1, 4],
    "min_samples_split": [2, 10],
    "n_estimators": [200, 1000]}
   ```
  Can you beat the baseline, or the linear regression, or best polynomial model with the best optimized Random Forest Regressor in terms of $R^2$ and root mean square error?

The implementation of the random forest regressor with grid search is in the function rforest_predict.
The test classification results are as follows:
R2:  0.9866
MSE: 1.0251

Although this is a good classification model, it performs slightly worse that multi regression in terms of MSE and slightly better than Multi regression in terms of R2. This model well out performs the baseline but is substantially beat by the polynomial regression model.


## References
The data was provided with this assignment and much of the code was adopted from the module 5 notebooks, however I referenced quite a bit the Scikit learn documentation.

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
https://scikit-learn.org/

Many places in the code where reference documentation is used a link is provided inline.

## Full code output
The following is the full output of the script, ./hw5.py

```
Baseline prediction
R2:  0.3147
MSE: 52.5047

Multi regression prediction
R2:  0.9883
MSE: 0.8977

Polynomial regression

Best params:  {'polynomialfeatures__degree': 3}
R2:  1.0000
MSE: 0.0002

Random forest regression
R2:  0.9857
MSE: 1.0921
```
