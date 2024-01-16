# Kaggle - House Prices Project

This project deals with the house price prediction using the dataset obtained from Kaggle https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


## Missing values computation

There were some columns with missing values. Imputation was performed on a few columns where the missing value does not have any meaning
As there were more categorical variables that require imputations, categorical variables were imputed first. This was performed using a tree-based algorithm (HistGradientBoostingClassifier)
Following which, the continuous variable(s) were imputed using support vector regression
