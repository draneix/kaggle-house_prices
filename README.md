# Kaggle - House Prices Project

This project deals with the house price prediction using the dataset obtained from Kaggle https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


## Missing values computation

There were some columns with missing values. Imputation was performed on a few columns where the missing value does not have any meaning
As there were more categorical variables that require imputations, categorical variables were imputed first. This was performed using a tree-based algorithm (HistGradientBoostingClassifier)
Following which, the continuous variable(s) were imputed using support vector regression

## Modelling
A multilayer perceptron is used to model the sale price of the flat.
The loss function used for RMSE.
Hyperparameter tuning was performed using Optuna:
    - Learning rate - [1e-4, 1e-2]
    - Batch size - [2 ** 4, 2 ** 6]
    - Optimiser - [Adam, SGD]
    - Number of layer - [1, 4]
    - Number of neurons in each layer - [1, 3000]
    - Activation function - [ReLU, Tanh, Sigmoid]
    - Dropout probability - [0.1, 0.5]
    - Additional dropout or batch norm layers - [True, False]