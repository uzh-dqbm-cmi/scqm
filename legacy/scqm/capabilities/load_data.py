import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
import pandas as pd
import sys


# Importing the  data and converting into the pandas dataframe
def load_raw_data(path: str) -> pd.DataFrame:
    robjects.r['load'](path)
    pandas2ri.activate()

    return robjects.r['l.tables']


# Importing the excel file to in/exclude columns based on their relevance
def load_excel_file(path: str) -> list():
    sheets = ['Patient','Visit','Medication','Health_issues', 'Mnyc_scoring',
              'Ratingen_scoring','Sonar_as','Sonar_ra','asas','basfi','basdai',
              'dlqi','euroquol','haq','psada','radai5','sf_12','socioeco']
    excel_files = list()
    for i in sheets:
        excel_file = pd.read_excel('/opt/data/SCQM_data_tables.xlsx', sheet_name=i, header=2)
        excel_files.append(excel_file)

    return excel_files


def filter_relevant_columns(table: pd.DataFrame, relevant_columns: pd.DataFrame):
    # Extract columns indicated by 1 in the excel sheet
    columns_to_be_included = relevant_columns[relevant_columns['Include'] == 1].iloc[:, 0].to_list()
    filtered_table = table[columns_to_be_included]

    return filtered_table


def remove_unique_value_columns(table: pd.DataFrame):
    # Remove columns which has only unique value
    filtered_table = table[[c for c in list(table) if len(table[c].unique()) > 1]]

    return filtered_table


# Function to visualize progress
def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


def extract_columns(data_tables, excel_files) -> dict():

    table_ids = ['p', 'v', 'm', 'hi', 'mny', 'rau', 'soa', 'sor', 'as',
                 'bf', 'bd', 'd', 'eq', 'h', 'ps', 'ra', 'sf', 'se']
    tables = dict()
    for i in range(len(data_tables)):
        progress(i, len(data_tables), 'Formatting table: ' + table_ids[i])
        df = filter_relevant_columns(robjects.r['l.tables'][i], excel_files[i])
        df = remove_unique_value_columns(df)
        tables[table_ids[i]] = df

    progress(i + 1, len(robjects.r['l.tables']), 'Tables are formatted')

    return tables
