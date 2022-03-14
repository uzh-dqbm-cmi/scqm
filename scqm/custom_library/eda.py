import os
from tkinter.tix import InputOnly
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import re

def find_regex_columns(df_dict, regex):
    """find which tables contain a given regex in their columns"""
    columns = [df.filter(regex=(regex)).columns for index, df in df_dict.items()]
    columns = [item for sublist in columns for item in sublist]
    columns_dict = {column : [index for index, table in df_dict.items() if column in table.columns] for column in columns}
    return columns_dict


def analyze_patient_subsets(df_dict, verbose=False):
    # find unique patients
    patient_dict = {}
    visit_dict = {}
    for index, table in df_dict.items():
        if 'patient_id' in table.columns:
            patient_dict[index] = set(table['patient_id'].unique())
        if 'uid_num' in table.columns:
            visit_dict[index] = set(table['uid_num'].unique())
    # find intersections
    for index in ['medications', 'visits']:
        for index_2 in ['visits', 'patients']:
            if index == index_2:
                pass
            elif verbose:
                print(
                    f'tables {index} and {index_2} length of patient intersection {len(patient_dict[index].intersection(patient_dict[index_2]))}')
    if verbose:
        print(
            f"total length of subsets visits {len(patient_dict['visits'])}, medications {len(patient_dict['medications'])} patients {len(patient_dict['patients'])}")
    # all patients in visits are in patients and all patients in medications are in patients
    # only 2 patients are in patients and not in visits --> dropdd
    # only 1 patient is in medications and not in visits --> drop
    if verbose:
        for index in list(df_dict.keys()):
            print(
                f"tables patients and {index} percentage of patient intersection {np.float(len(patient_dict[index].intersection(patient_dict['patients']))/len(patient_dict[index]))}")
        print(f'all patients in other tables are in patients')

    return patient_dict


def link_patients_visits(df_dict):
    def get_interval(x):
        #compute medication [start, end] interval
        # check if nat
        #TODO implement e.g. empty intervals for that
        if x.isna().values.any():
            return []
        elif x[1] < x[0]:
            return []
        return pd.Interval(x[0], x[1], closed='both')
    # link between visit_id, patients and medications
    tables_with_visit = []
    for index, table in df_dict.items():
        if 'uid_num' in table.columns:
            tables_with_visit.append(index)
    print(f'out of {len(df_dict)} tables {len(tables_with_visit)} contain a visit_id')
    # link patient id and visits
    grouped_visit_df = df_dict['visits'].groupby('patient_id')
    patients_visits = {elem: list(grouped_visit_df.get_group(elem)['uid_num'].values) if elem in df_dict['visits']['patient_id'].unique(
    ) else [] for elem in df_dict['patients']['patient_id'].unique()}
    num_of_visits = np.array([len(elem) for index, elem in patients_visits.items()])
    print(
        f'Min number of visits {num_of_visits.min()}, max number of visits {num_of_visits.max()}, mean number of visits {num_of_visits.mean()}')
    # create new columns for medication intervals
    df_dict['medications']['start_end'] = df_dict['medications'][[
        'medication_start_date', 'medication_end_date']].apply(get_interval, axis=1)
    grouped_med_df = df_dict['medications'].groupby('patient_id')
    num_of_medications = grouped_med_df.apply(len)
    print(f'Min number of medications {num_of_medications.min()}, max num of medications {num_of_medications.max()} mean number of medications {num_of_medications.mean()}')

    # patient dict with visits and medications
    patient_dict = {patient: {'visit_times': grouped_visit_df.get_group(
        patient)['visit_date'].values if patient in df_dict['visits']['patient_id'].unique() else [], 
        'visit_id': grouped_visit_df.get_group(
        patient)['uid_num'].values if patient in df_dict['visits']['patient_id'].unique() else [],
        'medication_intervals': grouped_med_df.get_group(patient)['start_end'].values if patient in df_dict['medications']['patient_id'].unique() else []} for patient in df_dict['patients']['patient_id'].unique()}
    
    return patients_visits, patient_dict

def eda_adanet_feature(gen_df, md_df, v_df):
    #age, disease_duration columns
    general_df = gen_df.copy()
    med_df = md_df.copy()
    visits_df = v_df.copy()
    general_df['age'] = (pd.to_datetime("today") - general_df['date_of_birth']) / np.timedelta64(1, 'Y')
    general_df['disease_duration'] = (pd.to_datetime("today") - general_df['date_first_symptoms']) / np.timedelta64(1, 'Y')
    general_df.drop(columns = ['date_of_birth', 'date_first_symptoms', 'date_diagnosis', 'patient_id'], inplace=True)
    med_df.drop(columns = ['patient_id', 'medication_start_date', 'medication_end_date'], inplace=True)
    visits_df.drop(columns = ['uid_num', 'patient_id'], inplace=True)
    # perc missing values and means
    means = pd.Series([])
    missing_values = pd.Series([])
    stds = pd.Series([])
    for df in [general_df, med_df, visits_df]:
        missing_values = pd.concat([missing_values, df.isna().sum()/len(df)])
        means = pd.concat([means, df.mean()])
        stds = pd.concat([stds, df.std()])
    # perc for categorical
    categories = {}
    for df in [general_df, med_df, visits_df]:
        for column in df.columns:
            if column not in means.index and column not in ['med_id', 'visit_date']:
                categories[column] = (df[column].value_counts()/len(df)).to_dict()
    categories = pd.concat({k: pd.DataFrame(v, index=[0]).T for k, v in categories.items()}, axis=0)
    means = pd.concat([means, stds], axis = 1).rename({0: 'means', 1: 'stds'})

    return missing_values, means, categories





