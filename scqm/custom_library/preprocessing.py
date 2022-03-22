
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime

#TODO in preprocessing medication_drug_classification is sometimes missing but we know it
def load_dfs():
    # load all tables
    data_path = '/opt/data/01_raw'
    tables = [f for f in listdir(data_path)]
    df_dict = {}
    for elem in tables:
        if 'csv' in elem:
            df_dict[elem[:-4]] = pd.read_csv(data_path + ('/') + elem).drop_duplicates()
        else:
            pass
            #df_dict[elem] = pd.read_excel(data_path + ('/') +elem)
    for index, table in df_dict.items():

        print(f'name {index} shape {table.shape}')
        if df_dict[index].filter(regex=('patient_id')).shape[1] == 1:
            # for consistency
            df_dict[index] = df_dict[index].rename(columns=lambda x: re.sub('.*patient_id', 'patient_id', x))
        else:
            print(f'table {index} has not ids for patients')
        if 'uid_num' or 'patient_id' in table.columns:
            pass
        else:
            print('PROBLEM')

    return df_dict


def preprocessing(df_dict, nan_prop=1):
    df_dict_processed = df_dict.copy()
    #TODO exclude columns with only single value
    #drop the two patients in patients that have no 
    # drop columns with nan proportion equal or more than nan_prop
    for index, table in df_dict_processed.items():
        # thresh : require that many non na values to keep the column
        df_dict_processed[index] = df_dict_processed[index].dropna(axis=1, thresh=int(len(table) * (1-nan_prop)+1))
        #print(f'kept {df_dict_processed[index].shape[1]} columns out of {df_dict[index].shape[1]}')
    # convert string dates to datetime and replace "unknown" by np.nan
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(pd.to_datetime)
        df_dict_processed[index] = df_dict_processed[index].replace('unknown', np.nan)
    # radai5 df has some missing visit ids drop these rows for now
    df_dict_processed['radai5'] = df_dict_processed['radai5'].dropna(subset=['uid_num'])
    # socioeco specific preprocessing
    df_dict_processed['socioeco']['.smoker'] = df_dict_processed['socioeco']['.smoker'].replace({'never_been_smoking':'i_have_never_smoked',
    'smoking_currently' : 'i_am_currently_smoking', 'a_former_smoker': 'i_am_a_former_smoker_for_more_than_a_year'})
    # medications specific preprocessing
    # sometimes written bsDMARD or brDMARD for the same thing
    df_dict_processed['medications']['medication_drug_classification'] = df_dict_processed['medications']['medication_drug_classification'].replace(
        {'bsDMARD': 'bDMARD', 'brDMARD': 'bDMARD'})
    df_dict_processed['medications']['medication_generic_drug'] = df_dict_processed['medications']['medication_generic_drug'].replace({'prednison_steroid_mr':'prednisone'})
    # add med_id identifiers to medications df
    df_dict_processed['medications']['med_id'] = ['med_' + str(i) for i in range(len(df_dict_processed['medications']))]
                
    return df_dict_processed


def extract_adanet_features(df_dict, transform_meds = True, das28=True, joint_df = False):
    general_df = df_dict['patients'][['patient_id', 'date_of_birth', 'gender', 'anti_ccp', 'ra_crit_rheumatoid_factor',
                                      'date_first_symptoms', 'date_diagnosis']]
    med_df = df_dict['medications'][['patient_id', 'med_id', 'medication_generic_drug', 'medication_drug_classification', 'medication_dose',
                                    'medication_start_date', 'medication_end_date']]
    visits_df = df_dict['visits'][['patient_id', 'uid_num', 'visit_date', 'weight_kg', 'das283bsr_score', 'n_swollen_joints', 'n_painfull_joints', 'bsr',
                                   'n_painfull_joints_28', 'height_cm', 'crp']]
    socioeco_df = df_dict['socioeco'][['uid_num', '.smoker']]
    radai_df = df_dict['radai5'][['uid_num', '.pain_level_today_radai', '.morning_stiffness_duration_radai',
                                  '.activity_of_rheumatic_disease_today_radai']]
    haq_df = df_dict['haq'][['uid_num', 'haq_score']]
    # keep only some specific medications and change the label of remaining to "other"
    drugs_to_keep = ['methotrexate', 'prednisone', 'rituximab', 'adalimumab', 'sulfasalazine', 'leflunomide', 'etanercept', 'infliximab']
    med_df.loc[~med_df["medication_generic_drug"].isin(
        drugs_to_keep), "medication_generic_drug"] = "Other"
    for other_df in [socioeco_df, radai_df, haq_df]:
        visits_df = visits_df.merge(other_df, how='outer', on='uid_num')
    if das28:
        #keep only visits with available das28 score
        print(f'Dropping {visits_df["das283bsr_score"].isna().sum()} out of {len(visits_df)} visits because das28 is unavailable')
        visits_df = visits_df.dropna(subset=['das283bsr_score'])
        # add column for classification
        visits_df['das28_category'] = [0 if value <= 2.6 else 1 for value in visits_df['das283bsr_score']]
        #keep only subset of patients appearing in visits_id
        patients = visits_df['patient_id'].unique()
        print(
            f'Keeping {len(general_df[general_df.patient_id.isin(patients)])} patients out of {len(general_df.patient_id.unique())}')
        general_df = general_df[general_df.patient_id.isin(patients)]
        med_df = med_df[med_df.patient_id.isin(patients)]
    #sort dfs
    visits_df.sort_values(['patient_id', 'visit_date'], inplace=True)
    general_df.sort_values(['patient_id'], inplace=True)
    med_df.sort_values(['patient_id', 'medication_start_date', 'medication_end_date'], inplace=True)
    targets_df = visits_df[['patient_id', 'visit_date', 'uid_num', 'das283bsr_score', 'das28_category']]

    if transform_meds:
        # add new column to med_df indicating for each event if it is a start or end of medication (0 false, 1 true) and replace med_start and med_end 
        # by unique column (date). If start date is not available, drop the row. If start and end are available duplicate the row (with different date and is_start dates)
        med_df = med_df.dropna(subset = ['medication_start_date'])
        med_df = med_df.rename({'medication_start_date': 'date'}, axis=1)
        med_df['is_start'] = 1
        tmp = med_df[med_df.medication_end_date.notna()].copy()
        tmp['date'] = tmp['medication_end_date']
        tmp['is_start'] = 0
        med_df = pd.concat([med_df, tmp]).drop(columns = ['medication_end_date'])
        med_df.sort_values(['patient_id', 'date'], inplace=True)
    #create a single df that contains all the info (for the baselines)
    if joint_df:
        visits_df['is_visit'] = 1
        med_df['is_visit'] = 0
        joint_df = pd.concat([visits_df.rename(
            columns={'visit_date': 'date'}), med_df], ignore_index=True).sort_values(by=['patient_id', 'date', 'uid_num'], axis=0)

    else: 
        joint_df = []
    return general_df, med_df, visits_df, targets_df, joint_df

# 
