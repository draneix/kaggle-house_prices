
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

from config import list_cols_with_na


def read_data():

    # Reading data
    # Train
    df_train = pd.read_csv("./data/train.csv")
    # Test
    # Note that test set has no SalePrice, which is what you are going to predict
    # Therefore, you need to get your own "test" set from the train data to evaluate your model
    # The test set is evaluated onto kaggle
    df_test = pd.read_csv("./data/test.csv")

    # Rule based data cleaning
    clean_data(df_train)
    clean_data(df_test)

    # Imputation of categorical variables
    df_combine_temp = pd.concat([df_train, df_test]).drop(columns=["Id", "SalePrice"])
    df_combine_temp.to_csv("check.csv", index=False)
    # Impute categorical columns with their most frequent values
    # TODO: Impute categorical variables using tree algorithms
    temp_df = df_combine_temp.isna().sum()
    list_cat_cols = [i for i in df_combine_temp.columns if df_combine_temp[i].dtype == "O"]
    cat_cols_to_impute = list(temp_df.loc[temp_df > 0].index)
    cat_cols_to_impute.remove("LotFrontage")
    df_combine_temp_2 = df_combine_temp.drop(columns="LotFrontage")
    ct_cat = ColumnTransformer([("ordinal", OrdinalEncoder(), list_cat_cols)])
    pipe_cat = Pipeline([("preprocess", ct_cat),
                         ("tree_impute", IterativeImputer(estimator=HistGradientBoostingClassifier()))])
    df_combine_temp_2 = pipe_cat.fit_transform(df_combine_temp_2)

    # # Impute LotFrontage with SVR
    # df_combine_temp_2.loc[:, "LotFrontage"] = df_combine_temp["LotFrontage"]

    return df_combine_temp_2


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
    # Also put garageyearblt as 0
    df.loc[df["GarageType"].isna(), "GarageCars"] = 0
    df.loc[df["GarageType"].isna(), "GarageArea"] = 0
    df.loc[df["GarageType"].isna(), "GarageYrBlt"] = 0

    # Drop 2 rows of data where GarageYrBlt is hard to determine
    df = df.loc[~df["GarageYrBlt"].isna()]

    # Fill missing values with "empty" to represent missing values that have meaning
    df[list_cols_with_na].fillna("empty", inplace=True)

    return
