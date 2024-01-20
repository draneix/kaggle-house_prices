import random
import numpy as np
import torch

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingClassifier

from config import list_cols_with_na


def read_data():
    """Read train and test data
    Clean train and test data
    Impute train and test data by combining them
    TODO: Clean up or loop imputation portion

    Returns:
        df_complete_temp_final (dataframe): combined train and test data after imputation
    """

    # Reading data
    # Train
    df_train = pd.read_csv("./data/train.csv")
    df_train.loc[:, "SalePrice"] = np.log(df_train["SalePrice"])
    # Test
    # Note that test set has no SalePrice, which is what you are going to predict
    # Therefore, you need to get your own "test" set from the train data to evaluate your model
    # The test set is evaluated onto kaggle
    df_test = pd.read_csv("./data/test.csv")

    # Rule based data cleaning
    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    df_combine = pd.concat([df_train, df_test]).reset_index(drop=True)
    df_combine_temp = df_combine.drop(columns=["Id", "SalePrice"])

    # Get categorical columns which will be ordinally encoded
    list_cat_cols = [i for i in df_combine_temp.columns if df_combine_temp[i].dtype == "O"]

    # Drop lotfrontage first, impute categorical variables first
    # Imputation of categorical variables using HistGradientBoostingClassifier, based on LightGBM LightGBM
    df_combine_temp_2 = df_combine_temp.drop(columns="LotFrontage")
    ord_encoder = OrdinalEncoder()
    tree_impute = IterativeImputer(estimator=HistGradientBoostingClassifier(), skip_complete=True)
    df_combine_temp_2[list_cat_cols] = ord_encoder.fit_transform(df_combine_temp_2[list_cat_cols])
    df_combine_temp_2 = tree_impute.fit_transform(df_combine_temp_2)

    # Get back dataframe and insert LotFrontage to be imputed next
    df_combine_temp_2 = pd.DataFrame((df_combine_temp_2), columns=tree_impute.get_feature_names_out())
    df_combine_temp_2[list_cat_cols] = ord_encoder.inverse_transform(df_combine_temp_2[list_cat_cols])
    df_combine_temp_2.loc[:, "LotFrontage"] = df_combine_temp["LotFrontage"]

    # Impute comtinuous columns using SVR
    ord_encoder = OrdinalEncoder()
    cont_imp = IterativeImputer(estimator=SVR(), skip_complete=True)
    df_combine_temp_2[list_cat_cols] = ord_encoder.fit_transform(df_combine_temp_2[list_cat_cols])
    df_combine_temp_2 = cont_imp.fit_transform(df_combine_temp_2)

    # Get final dataset
    df_combine_temp_final = pd.DataFrame(df_combine_temp_2, columns=cont_imp.get_feature_names_out())
    df_combine_temp_final[list_cat_cols] = ord_encoder.inverse_transform(df_combine_temp_final[list_cat_cols])

    # Add back SalePrice
    df_combine_temp_final.loc[:, "SalePrice"] = df_combine["SalePrice"]

    # Get train and test dataset based on length
    df_train_final = df_combine_temp_final.iloc[:len(df_train)]
    df_test_final = df_combine_temp_final.iloc[len(df_train):]
    df_test_final.drop(columns="SalePrice", inplace=True)

    return df_train_final, df_test_final


def clean_data(df):
    """Function to clean dataframe based on findings in data_cleaning.ipynb

    Args:
        df (dataframe): train or test dataframe
    """

    # If BsmtQual is NA, SF must be zero, BsmtFullBath and BsmtHalfBath must be zero too
    df.loc[df["BsmtQual"].isna(), "BsmtFinSF1"] = 0
    df.loc[df["BsmtQual"].isna(), "BsmtFinSF2"] = 0
    df.loc[df["BsmtQual"].isna(), "TotalBsmtSF"] = 0
    df.loc[df["BsmtQual"].isna(), "BsmtUnfSF"] = 0
    df.loc[df["BsmtQual"].isna(), "BsmtFullBath"] = 0
    df.loc[df["BsmtQual"].isna(), "BsmtHalfBath"] = 0

    # If MasVnrType is NA, MasVnrArea should be zero
    df.loc[df["MasVnrType"].isna(), "MasVnrArea"] = 0

    # If GarageType is NA, GarageCars and GarageArea must be zero
    df.loc[df["GarageType"].isna(), "GarageCars"] = 0
    df.loc[df["GarageType"].isna(), "GarageArea"] = 0
    # Also put garageyearblt as 0
    df.loc[df["GarageType"].isna(), "GarageYrBlt"] = 0
    # Set GarageYrBlt as mean
    df.loc[df["GarageYrBlt"].isna(), "GarageYrBlt"] = df.loc[df["GarageYrBlt"] != 0]["GarageYrBlt"].mean()

    # Fill missing values with "empty" to represent missing values that have meaning
    df.loc[:, list_cols_with_na] = df[list_cols_with_na].fillna("empty")

    return df


def seed_everything(SEED_NUM=12):

    np.random.seed(SEED_NUM)
    random.seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)
    return
