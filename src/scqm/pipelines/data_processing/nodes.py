"""
Defines all nodes potentially usable by ....
"""

import pandas as pd

def remove_unique_value_columns(table: pd.DataFrame):
    """Remove unique columns from a pandas dataframes.

    Args:
        table: pandas data frame
    Returns:
        filtered_table:  dataframe with removed unique value columns
    """

    filtered_table = table[[c for c in list(table) if len(table[c].unique()) > 1]]

    return filtered_table

def filter_relevant_columns(table: pd.DataFrame, relevant_columns: pd.DataFrame):
    """Remove unique columns from a pandas dataframes.

    Args:
        table: pandas data frame
    Returns:
        filtered_table:  dataframe with relevant columns
    """
    columns_to_be_included = relevant_columns[relevant_columns['Include'] == 1].iloc[:, 0].to_list()
    filtered_table = table[columns_to_be_included]

    return filtered_table