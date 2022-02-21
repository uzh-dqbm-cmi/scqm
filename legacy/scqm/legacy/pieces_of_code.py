import pandas as pd

# IMPORT excel file
sheets = ['Patient','Visit','Medication','Health_issues', 'Mnyc_scoring',
          'Ratingen_scoring','Sonar_as','Sonar_ra','asas','basfi','basdai',
          'dlqi','euroquol','haq','psada','radai5','sf_12','socioeco']
excel_files = list()
for i in sheets:
    excel_file = pd.read_excel('/opt/data/SCQM_data_tables.xlsx', sheet_name=i, header = 2)
    excel_files.append(excel_file)


def filter_relevant_columns(table: pd.DataFrame, relevant_columns: pd.DataFrame):
    # Extract columns indicated by 1 in the excel sheet
    columns_to_be_included = relevant_columns[relevant_columns['Include'] == 1].iloc[:, 0].to_list()
    filtered_table = table[columns_to_be_included]

    return filtered_table


def remove_unique_value_columns(table: pd.DataFrame):
    # Remove columns which has only unique value
    filtered_table = table[[c for c in list(table) if len(table[c].unique()) > 1]]

    return filtered_table

