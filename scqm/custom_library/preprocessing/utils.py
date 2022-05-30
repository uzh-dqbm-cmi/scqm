import numpy as np
import pandas as pd
from typing import Tuple


def map_category(df: pd.DataFrame, column: str) -> Tuple[pd.Series, dict]:
    """Map pandas series with categoris (as strings) to integer categories
    (only to be used when one hot encoding is not convenient)

    Args:
        df (pd.DataFrame): initial df
        column (str): column to convert

    Returns:
        Tuple[pd.Series, dict]: transformed series and dictionnary with mapping
    """
    mapping = {
        key: value for value, key in enumerate(sorted(df[column].dropna().unique()))
    }
    new_column = df[column].replace(mapping)
    return new_column, mapping


def das28_increase(df: pd.DataFrame) -> pd.DataFrame:
    """Find increasing or decreasing das28 values with respect to previous visit.

    Compute additional column with 0 if the das28 value is stable since last visit (i.e. delta das28 <= 1.2, 1 if there is an increase and 2 if there is a decrease)
    Args:
        df (pd.DataFrame): dataframe with das28 values

    Returns:
        pd.DataFrame: df with additional column.
    """
    # 0 if stable (i.e. delta das28 <= 1.2, 1 if increase else 2 if decrease)
    df["das28_increase"] = [
        np.nan
        if index == 0
        else 0
        if abs(
            df["das283bsr_score"].iloc[index - 1] - df["das283bsr_score"].iloc[index]
        )
        <= 1.2
        else 1
        if df["das283bsr_score"].iloc[index - 1] < df["das283bsr_score"].iloc[index]
        else 2
        for index in range(len(df))
    ]
    return df


def get_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Change categorical columns of df to one-hot encoding

    Args:
        df (pd.DataFrame): df to convert

    Returns:
        pd.DataFrame: modified df with one-hot encoded columns
    """
    columns = [
        col
        for col in df.columns
        if df[col].dtype == "object" and df[col].nunique() < 10
    ]
    df_dummies = pd.get_dummies(
        df, columns=columns, dummy_na=True, drop_first=True, prefix_sep="_dummy_"
    )
    # to put nan values back in columns instead of having a dedicated column
    nan_df = df_dummies.loc[:, df_dummies.columns.str.endswith("_nan")]

    for col_nan in nan_df.columns:
        col_id = col_nan.split("_dummy_")[0]
        targets = df_dummies.columns[
            df_dummies.columns.str.startswith(col_id + "_dummy_")
        ]
        index = df_dummies[df_dummies[col_nan] == 1].index
        df_dummies.loc[index, targets] = np.nan
    df_dummies.drop(
        df_dummies.columns[df_dummies.columns.str.endswith("_nan")],
        axis=1,
        inplace=True,
    )

    return df_dummies
