# Kaggle - House Prices Project

This project deals with the house price prediction using the dataset obtained from Kaggle https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


## Missing values computation

There were some columns with missing values. Imputation was performed on a few columns where the missing value does not have any meaning
- LotFrontage - imputed through mean, median or MICE (TBC)
- Electrical - imputed through the most common category
- BsmtFinType2 - imputed through similar value in BsmtFinType1 and BsmtFinSF1
- BsmtExposure - imputed as "No"
- MasVnrArea - imputed as 0 if MasVnrType is NA
- MasVnrType - imputed through similar values after grouping for exterior1st, exterior2nd, exterqual, ExterCond

