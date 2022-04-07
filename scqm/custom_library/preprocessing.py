
import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime
import pickle

#TODO in preprocessing medication_drug_classification is sometimes missing but we know it


def load_dfs_all_data(subset = None):
    # load all tables
    data_path = '/opt/data/01_raw_scqm_all/SCQM_data_tables_all_data_reduced.pickle'
    with open(data_path, 'rb') as handle:
        df_dict = pickle.load(handle)
    
    # rename dfs to match names in previous dataset
    name_mapping = {'p': 'patients', 'v': 'visits', 'm': 'medications', 'hi': 'healthissues', 'mny': 'modifiednewyorkxrayscore', 'rau': 'ratingenscore',
    'soa': 'sonaras', 'sor': 'sonarra', 'as': 'asas','bf': 'basfi', 'bd': 'basdai', 'd': 'dlqi', 'eq': 'euroquol', 'h': 'haq', 'ps': 'psada', 'ra': 'radai5',
    'sf': 'sf_12', 'se': 'socioeco'}
    for key in list(df_dict.keys()):
        df_dict[name_mapping[key]] = df_dict.pop(key)
    for key in df_dict:
        df_dict[key] = df_dict[key].rename(columns=lambda x: re.sub('^[^.]*', '', x)[1:])

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
    if subset is not None:
        # keep only a certain percentage of the whole data
        patients_to_keep = np.random.choice(df_dict['patients']['patient_id'].unique(), size=int(
            subset * len(df_dict['patients']['patient_id'].unique())), replace=False)
        for key in df_dict:
            df_dict[key] = df_dict[key][df_dict[key].patient_id.isin(patients_to_keep)]

    return df_dict
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

def clean_dates(df_dict):
    df_dict_processed = df_dict.copy()
    #some specific outlier preprocessing :
    df_dict_processed['patients']['date_of_birth'][df_dict_processed['patients']['date_of_birth'] == '1063-05'] = '1963-05'
    df_dict_processed['visits']['date_of_birth'][df_dict_processed['visits']
                                                ['date_of_birth'] == '1063-05'] = '1963-05'
    df_dict_processed['visits']['osteodensitometrie_date'][df_dict_processed['visits']
                                                        ['osteodensitometrie_date'] == '2913-03-01'] = '2013-03-01'
    df_dict_processed['visits']['osteodensitometrie_date'] = df_dict_processed['visits']['osteodensitometrie_date'].apply(
        lambda x: '2013-' + x.split('-')[1] + '-' + x.split('-')[2] if x is not None and x.split('-')[0] == '2301' else x)
    df_dict_processed['visits']['hospital_rehab_stay_due_to_arthritis_date'][df_dict_processed['visits']
                                                                            ['hospital_rehab_stay_due_to_arthritis_date'] == '3013-11-01'] = '2013-11-01'
    df_dict_processed['patients']['mnyc_date_positive'] = df_dict_processed['patients']['mnyc_date_positive'].apply(
        lambda x: x.split('|')[0] if x is not None else x)
    df_dict_processed['visits']['mnyc_date_positive'] = df_dict_processed['visits']['mnyc_date_positive'].apply(
        lambda x: x.split('|')[0] if x is not None else x)
    df_dict_processed['medications']['medication_start_date'][df_dict_processed['medications']
                                                            ['medication_start_date'] == '3021-03-15'] = '2021-03-15'
    df_dict_processed['medications']['stored_start_date'][df_dict_processed['medications']
                                                        ['stored_start_date'] == '3021-03-15'] = '2021-03-15'
    df_dict_processed['healthissues']['health_issue_date'][df_dict_processed['healthissues']
                                                        ['health_issue_date'] == '20'] = None
    return df_dict_processed
def preprocessing(df_dict, nan_prop=1):
    df_dict_processed = df_dict.copy()
    #drop the two patients in patients that have no 
    # drop columns with nan proportion equal or more than nan_prop
    for index, table in df_dict_processed.items():
        # thresh : require that many non na values to keep the column
        df_dict_processed[index] = df_dict_processed[index].dropna(axis=1, thresh=int(len(table) * (1-nan_prop)+1))
        #print(f'kept {df_dict_processed[index].shape[1]} columns out of {df_dict[index].shape[1]}')
    #some specific outlier preprocessing :
    df_dict_processed = clean_dates(df_dict_processed)
    #other specific preprocessing
    df_dict_processed['visits']['crp'] = pd.to_numeric(df_dict_processed['visits']['crp'].apply(
        lambda x: x.split('< ')[1] if (x is not None and len(x.split('< ')) > 1) else x))
    # convert string dates to datetime and replace "unknown" by np.nan
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(pd.to_datetime)
        df_dict_processed[index] = df_dict_processed[index].replace('unknown', np.nan)
    # unify date column name
    for index, elem in df_dict_processed.items():
        name = list(elem.filter(regex="visit_date").columns)
        if len(name) == 1:
            df_dict_processed[index] = elem.rename(columns={name[0]: 'date'})
    # manually change date names in ratingenscore, healthissues and modifiednewyorkxrayscore
    df_dict_processed['ratingenscore'] = df_dict_processed['ratingenscore'].rename(
        columns={'imaging_score_scoring_date': 'date'})
    # for healthissues, '.health_issue_date' is 90% missing, use recording time when it is missing ?
    df_dict_processed['modifiednewyorkxrayscore'] = df_dict_processed['modifiednewyorkxrayscore'].rename(
        columns={'imaging_score_scoring_date': 'date'})
    # drop columns with unique values
    for index, table in df_dict_processed.items():
        for col in table.columns:
            if table[col].nunique() == 1:
                df_dict_processed[index] = df_dict_processed[index].drop(col, axis=1)
    # radai5 df has some missing visit ids drop these rows for now (#TODO not forget to change)
    df_dict_processed['radai5'] = df_dict_processed['radai5'].dropna(subset=['uid_num'])
    # socioeco specific preprocessing
    df_dict_processed['socioeco']['smoker'] = df_dict_processed['socioeco']['smoker'].replace({'never_been_smoking':'i_have_never_smoked',
    'smoking_currently' : 'i_am_currently_smoking', 'a_former_smoker': 'i_am_a_former_smoker_for_more_than_a_year'})
    # medications specific preprocessing
    # sometimes written bsDMARD or brDMARD for the same thing
    df_dict_processed['medications']['medication_drug_classification'] = df_dict_processed['medications']['medication_drug_classification'].replace(
        {'bsDMARD': 'bDMARD', 'brDMARD': 'bDMARD'})
    df_dict_processed['medications']['medication_generic_drug'] = df_dict_processed['medications']['medication_generic_drug'].replace({'prednison_steroid_mr':'prednisone'})
    # add med_id identifiers to medications df
    df_dict_processed['medications']['med_id'] = ['med_' + str(i) for i in range(len(df_dict_processed['medications']))]

    df_dict_processed['medications'] = find_drug_categories_and_names(df_dict_processed['medications'])
    # unify the different visit_date columns and rename to date

    return df_dict_processed


def das28_increase(df):
    # 0 if stable (i.e. delta das28 <= 1.2, 1 if increase else 2 if decrease)
    df["das28_increase"] = [np.nan if index == 0 else 0 if abs(df['das283bsr_score'].iloc[index - 1] - df['das283bsr_score'].iloc[index]) <= 1.2
                            else 1 if df['das283bsr_score'].iloc[index - 1]
                            < df['das283bsr_score'].iloc[index] else 2 for index in range(len(df))]
    return df

def extract_adanet_features(df_dict, transform_meds = True, das28=True, only_meds=False, joint_df = False):
    general_df = df_dict['patients'][['patient_id', 'date_of_birth', 'gender', 'anti_ccp', 'ra_crit_rheumatoid_factor',
                                      'date_first_symptoms', 'date_diagnosis']]
    med_df = df_dict['medications'][['patient_id', 'med_id', 'medication_generic_drug', 'medication_drug_classification', 'medication_dose',
                                    'medication_start_date', 'medication_end_date']]
    visits_df = df_dict['visits'][['patient_id', 'uid_num', 'date', 'weight_kg', 'das283bsr_score', 'n_swollen_joints', 'n_painfull_joints', 'bsr',
                                   'n_painfull_joints_28', 'height_cm', 'crp']]
    socioeco_df = df_dict['socioeco'][['patient_id','uid_num', 'date', 'smoker']]
    radai_df = df_dict['radai5'][['patient_id','uid_num', 'date', 'pain_level_today_radai', 'morning_stiffness_duration_radai',
                                  'activity_of_rheumatic_disease_today_radai']]
    haq_df = df_dict['haq'][['patient_id', 'uid_num', 'date', 'haq_score']]
    # keep only some specific medications and change the label of remaining to "other"
    drugs_to_keep = ['methotrexate', 'prednisone', 'rituximab', 'adalimumab', 'sulfasalazine', 'leflunomide', 'etanercept', 'infliximab']
    med_df.loc[~med_df["medication_generic_drug"].isin(
        drugs_to_keep), "medication_generic_drug"] = "Other"
    # for other_df in [socioeco_df, radai_df, haq_df]:
    #     visits_df = visits_df.merge(other_df, how='outer', on='uid_num')
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
        # column saying if das28 at next visit increased (1) or decreased/remained stable 0
        visits_df['das28_increase'] = np.nan
        visits_df = visits_df.groupby('patient_id').apply(das28_increase)
        general_df = general_df[general_df.patient_id.isin(patients)]
        med_df = med_df[med_df.patient_id.isin(patients)]
        socioeco_df = socioeco_df[socioeco_df.patient_id.isin(patients)]
        radai_df = radai_df[radai_df.patient_id.isin(patients)]
        haq_df = haq_df[haq_df.patient_id.isin(patients)]

    if only_meds :
        patients = med_df['patient_id'].unique()
        print(
            f'keeping only patients with medical info, keeping {len(patients)} out of {len(general_df.patient_id.unique())}')
        general_df = general_df[general_df.patient_id.isin(patients)]
        visits_df = visits_df[visits_df.patient_id.isin(patients)]
        socioeco_df = socioeco_df[socioeco_df.patient_id.isin(patients)]
        radai_df = radai_df[radai_df.patient_id.isin(patients)]
        haq_df = haq_df[haq_df.patient_id.isin(patients)]

    #sort dfs
    visits_df.sort_values(['patient_id', 'date'], inplace=True)
    general_df.sort_values(['patient_id'], inplace=True)
    med_df.sort_values(['patient_id', 'medication_start_date', 'medication_end_date'], inplace=True)
    targets_df = visits_df[['patient_id', 'date', 'uid_num', 'das283bsr_score', 'das28_category', 'das28_increase']]
    socioeco_df.sort_values(['patient_id', 'date'], inplace =True)
    radai_df.sort_values(['patient_id', 'date'], inplace = True)
    haq_df.sort_values(['patient_id', 'date'], inplace = True)

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
        joint_df = pd.concat([visits_df, med_df], ignore_index=True).sort_values(by=['patient_id', 'date', 'uid_num'], axis=0)

    else: 
        joint_df = []
    return general_df, med_df, visits_df, targets_df, socioeco_df, radai_df, haq_df, joint_df

def find_drug_categories_and_names(df):
    """replace missing drug names and categories in df medications"""
    #TODO complete for new data
    # Reassmbly of the medication
    # Create a drug to category dictionary to impute any missing category values
    drugs_per_cat = df.groupby(by='medication_drug_classification')['medication_drug'].apply(list)
    drugs_per_generic = df.groupby(by='medication_generic_drug')['medication_drug'].apply(list)
    drug_to_category = dict()
    drug_to_generic = dict()

    for i in range(len(drugs_per_cat)):
        list_of_subcat_drug = list(set(drugs_per_cat[i]))
        for j in list_of_subcat_drug:
            drug_to_category[j] = drugs_per_cat.index[i]

    for i in range(len(drugs_per_generic)):
        list_of_drugs_in_genericdrug = list(set(drugs_per_generic[i]))
        for j in list_of_drugs_in_genericdrug:
            drug_to_generic[j] = drugs_per_generic.index[i]

    # Manually add the missing drugs:
    drug_to_category['spiricort'] = 'steroid'
    drug_to_category['Imurek (Azathioprin)'] = 'csDMARD'
    drug_to_generic['spiricort'] = 'spiricort'
    drug_to_generic['Imurek (Azathioprin)'] = 'azathioprine'


    # Display the medications without category
    not_cat_drugs = list(set(df['medication_drug'].to_list()) - set(list(drug_to_category.keys())))
    print(not_cat_drugs)

    # Display the medications without generic label
    no_generic_label = list(set(df['medication_drug'].to_list()) - set(list(drug_to_generic.keys())))
    print(no_generic_label)

    # Impute all missing medication category based on the created category dictionary
    for i in df.index:
        if pd.isnull(df['medication_drug_classification'][i]):
            if df['medication_drug'][i] not in not_cat_drugs:
                df['medication_drug_classification'][i] = drug_to_category[df['medication_drug'][i]]

    print(df['medication_drug_classification'].unique())
    # Impute all missing generic medication label based on the created generic label dictionary

    for i in df.index:
        if pd.isnull(df['medication_generic_drug'][i]):
            if df['medication_drug'][i] not in no_generic_label:
                df['medication_generic_drug'][i] = drug_to_generic[df['medication_drug'][i]]

    return df
