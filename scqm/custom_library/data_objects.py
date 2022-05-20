import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import Masks
import pickle
from preprocessing import get_dummies, map_category


class DataObject:
    def __init__(self, df_dict, patient_id):
        for name in df_dict.keys():
            setattr(self, str(name) + '_df', df_dict[name][df_dict[name]['patient_id'] == patient_id])
        return

    def add_info(self, name, value):
        setattr(self, name, value)
        return


class Patient(DataObject):
    # TODO be coherent with sorting (also other dfs)
    def __init__(self, df_dict, patient_id, event_names):
        super().__init__(df_dict, patient_id)
        self.id = patient_id
        self.visits = self.get_visits()
        self.medications = self.get_medications()
        self.event_names = event_names
        self.other_events = self.get_other_events()
        self.timeline = self.get_timeline()
        self.visits_to_predict = self.visits
        return

    def get_visits(self):
        self.visits_df = self.a_visit_df.sort_values(by='date', axis=0)
        self.visit_ids = self.visits_df['uid_num'].values
        self.visit_dates = self.visits_df['date'].values
        self.num_a_visit_events = len(self.visit_ids)
        visits = []
        for id_, date in zip(self.visit_ids, self.visit_dates):
            visits.append(Visit(self, id_, date))
        return visits

    def get_medications(self):
        self.medications_df = self.med_df.sort_values(by='date', axis=0)
        self.med_ids = self.medications_df['med_id'].unique()
        # total number of medication events counting all the available start and end dates
        self.num_med_events = len(self.medications_df['med_id'])
        self.med_intervals = []
        meds = []
        for id_ in self.med_ids:
            m = Medication(self, id_)
            meds.append(m)
            self.med_intervals.append(m.interval)
        return meds

    def get_other_events(self):
        other_events = []
        events = [name for name in self.event_names if name not in ['a_visit', 'med']]
        for name in events:
            df = getattr(self, name + '_df')
            setattr(self, 'num_' + name + '_events', len(df))
            for index in df.index:
                # TODO change maybe uid_num and put generic event id
                if 'uid_num' in df.columns:
                    other_events.append(Event(name, df.loc[index, 'uid_num'], df.loc[index, 'date']))
                else:
                    other_events.append(Event(name, date=df.loc[index, 'date']))
        return other_events

    def get_timeline(self):
        # get sorted dates of events
        visit_event_list = [(self.visit_dates[index], 'a_visit', self.visit_ids[index])
                            for index in range(len(self.visit_dates))]
        med_sart_list = [(self.med_intervals[index][0], 'med_s', self.med_ids[index])
                         for index in range(len(self.med_intervals))]
        med_end_list = [(self.med_intervals[index][1], 'med_e', self.med_ids[index])
                        for index in range(len(self.med_intervals))]
        other_events = [(event.date, event.name, event.id) for event in self.other_events]
        all_events = visit_event_list + med_sart_list + med_end_list + other_events
        # the 'a', 'b', 'c' flags before visit and meds are there to ensure that if they occur at the same date, the visit comes before
        # remove NaT events
        all_events = [event for event in all_events if not pd.isnull(event[0])]
        all_events.sort()
        # get sorted ids of events, format : True --> visit, False --> medication along with id
        # e.g. [(True, visit_id), (False, med_id), ...]
        self.timeline_mask = [(event[1], event[2])
                              if event[1] not in ['med_s', 'med_e'] else ('med', event[2]) for event in all_events]

        self.timeline_visual = ['v' if event[1] == 'a_visit' else 'm_s' if event[1] == 'med_s' else 'm_e' if event[1] == 'm_e' else event[1]
                                for event in all_events]
        self.mask = [[event[0]] for event in self.timeline_mask]

        return all_events

    def get_cropped_timeline(self, n=1, min_time_since_last_event=0, max_time_since_last_event=600):
        if n > len(self.visits):
            raise ValueError('n bigger than number of visits')
        # get all events up to n-th visit
        else:
            cropped_timeline = []
            num_of_each_event = torch.zeros(size=(len(self.event_names),))
            index_of_visit = self.event_names.index('a_visit')
            index = 0
            # date of n-th visit
            date_nth_visit = pd.Timestamp(self.visits[n - 1].date)
            # while number of visits < n and while ? other part is redundant no ?
            while (num_of_each_event[index_of_visit] < n and index < len(self.timeline) and (date_nth_visit - self.timeline[index][0]).days > min_time_since_last_event):
                event = self.timeline[index]
                cropped_timeline.append(event)
                if event[1] in ['med_s', 'med_e']:
                    index_of_event = self.event_names.index('med')
                else:
                    index_of_event = self.event_names.index(event[1])
                num_of_each_event[index_of_event] += 1
                index += 1
            time_to_next = (date_nth_visit - self.timeline[index-1][0]).days
    #         #remove last visit
    #         cropped_timeline = cropped_timeline[:-1]
    #         num_of_each_event[index_of_visit] -= 1
            # cropped_timeline_mask = [(event[1], event[2])
            #                          for event in cropped_timeline]
            cropped_timeline_mask = [(event[1], event[2])
                                     if event[1] not in ['med_s', 'med_e'] else ('med', event[2]) for event in cropped_timeline]
            cropped_timeline_visual = ['v' if event[1] == 'a_visit' else 'm_s' if event[1] == 'med_s' else 'm_e' if event[1] == 'med_e' else 'e'
                                       for event in cropped_timeline]
            to_predict = True if len(cropped_timeline) > 0 else False
            if to_predict and ((date_nth_visit - cropped_timeline[-1][0]).days > max_time_since_last_event):
                to_predict = False
            return num_of_each_event, cropped_timeline, cropped_timeline_mask, cropped_timeline_visual, to_predict


class Event:
    def __init__(self, name, id='no_id', date=None):
        self.name = name
        self.id = id
        self.date = date


class Visit(Event):
    def __init__(self, patient_class, visit_id, date, target='das283bsr_score'):
        super().__init__('a_visit', visit_id)
        self.patient = patient_class
        self.date = date
        self.data = self.patient.visits_df[self.patient.visits_df['uid_num'] == visit_id]
        if target:
            self.target = self.data[target]
        self.med_start = []
        self.med_during = []
        self.med_end = []
        return


class Medication(Event):
    def __init__(self, patient_class, med_id):
        super().__init__('med', med_id)
        self.patient = patient_class
        self.data = self.patient.medications_df[self.patient.medications_df['med_id'] == self.id]
        # start and end available
        if len(self.data) == 2:
            self.start_date = self.data[self.data['is_start'] == 1]['date'].item()
            self.end_date = self.data[self.data['is_start'] == 0]['date'].item()
        # only start available
        else:
            self.start_date = self.data['date'].item()
            self.end_date = pd.NaT
        self.interval = [self.start_date, self.end_date]
        # self.get_related_visits()
        # TODO implement flags for missing data
        self.flag = None
        return

    def get_related_visits(self, tol=0):
        self.visit_start = []
        self.visit_end = []
        self.visit_during = []
        for visit in self.patient.visits:
            # self.is_in_interval(visit)
            self.is_close_to_interval(visit, tol)
        return

    def is_close_to_interval(self, visit_class, tol):

        tol = datetime.timedelta(days=tol)
        if self.interval[0] - tol <= visit_class.date <= self.interval[0]:
            self.visit_start.append(visit_class.id)
            # TODO change this dependency
            visit_class.med_start.append(self.id)
        elif self.interval[1] - tol <= visit_class.date <= self.interval[1]:
            self.visit_end.append(visit_class.id)
            visit_class.med_end.append(self.id)
        elif self.interval[0] < visit_class.date < self.interval[1] - tol:
            self.visit_during.append(visit_class.id)
            visit_class.med_during.append(self.id)
        # all of the above conditions are False if at least one of the dates is NaT
        return

#mapping and get_item


class Dataset:
    def __init__(self, device, df_dict, ids, target_category_name, event_names, min_num_visits, mapping=None):
        self.initial_df_dict = df_dict
        self.patient_ids = list(ids)
        self.target_category_name = target_category_name
        self.event_names = event_names
        self.min_num_visits = min_num_visits
        self.device = device
        self.instantiate_patients()
        if mapping is not None:
            self.mapping = mapping
        # self.df_names = []
        # for name in df_dict.keys():
        #     name_df = str(name) + '_df'
        #     setattr(self, name_df, df_dict[name][df_dict[name]['patient_id'].isin(ids)])
        #     self.df_names.append(name_df)
        return

    def get_masks(self, min_time_since_last_event=30, max_time_since_last_event=450):
        print(f'Getting masks....')
        self.mapping_for_masks = {patient: index for index, patient in enumerate(self.patient_ids)}
        self.reverse_mapping_for_masks = {value: key for key, value in self.mapping_for_masks.items()}
        self.masks = Masks(self.device, self.patient_ids)
        self.masks.get_masks(self, debug_patient=None, min_time_since_last_event=min_time_since_last_event,
                             max_time_since_last_event=max_time_since_last_event)
        self.stratifier = {num_visit: [self.reverse_mapping_for_masks[patient_index] for patient_index in range(len(
            self.masks.num_visits)) if self.masks.num_visits[patient_index] == num_visit] for num_visit in range(1, max(self.masks.num_visits) + 1)}
        return

    def instantiate_patients(self):
        self.patients = {id_: Patient(self.initial_df_dict, id_, self.event_names) for id_ in tqdm(self.patient_ids)}

        return

    def drop(self, ids):
        # drop specific indices from dataset
        if not isinstance(ids, list):
            ids = list(ids)
        for id in ids:
            if id in self.patient_ids:
                del self.patients[id]
                self.patient_ids.remove(id)
        return

    def inclusion_criteria(self):
        # keep only the patients satisfying inclusion criteria
        with open('/opt/data/processed/patients_satisfying_inclusion_criteria.pickle', 'rb') as handle:
            patients_to_keep = pickle.load(handle)
        to_drop = list(set(self.patient_ids).difference(patients_to_keep))
        print(f'Dropping {len(to_drop)} patients because they dont satisfy inclusion criteria')
        self.drop(to_drop)
        print(f'{len(self)} patients remaining')
        return

    def create_dfs(self):
        self.df_names = []
        for name in self.initial_df_dict.keys():
            name_df = str(name) + '_df'
            setattr(self, name_df, self.initial_df_dict[name]
                    [self.initial_df_dict[name]['patient_id'].isin(self.patient_ids)])
            self.df_names.append(name_df)
        return

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, x):
        return self.patients[x]

    def move_to_device(self, device):
        for tensor_name in self.tensor_names:
            setattr(self, tensor_name, getattr(self, tensor_name).to(device))

        return

    def split_data(self, prop_valid=0.1, prop_test=0.1, stratify=True):
        # stratify on number of visits
        if stratify:
            self.train_ids = []
            self.valid_ids = []
            self.test_ids = []
            for num_visits in range(1, max(self.masks.num_visits) + 1):
                available_ids = self.stratifier[num_visits]
                length = len(available_ids)
                test_size = int(length * prop_test)
                valid_size = int(length * prop_valid)
                train_size = length - valid_size - test_size
                train_i = np.random.choice(available_ids, size=train_size, replace=False)
                self.train_ids.extend(train_i)
                available_ids = [id_ for id_ in available_ids if id_ not in train_i]
                valid_i = np.random.choice(available_ids, size=valid_size, replace=False)
                self.valid_ids.extend(valid_i)
                self.test_ids.extend([id_ for id_ in available_ids if id_ not in valid_i])
            self.train_ids = np.array(self.train_ids)
            self.valid_ids = np.array(self.valid_ids)
            self.test_ids = np.array(self.test_ids)
        else:
            test_size = int(len(self) * prop_test)
            valid_size = int(len(self) * prop_valid)
            train_size = len(self) - valid_size - test_size
            train_ids = np.random.choice(self.patient_ids, size=train_size, replace=False)
            # order
            self.train_ids = np.array([id_ for id_ in self.patient_ids if id_ in train_ids])
            available_ids = [id_ for id_ in self.patient_ids if id_ not in self.train_ids]
            valid_ids = np.random.choice(available_ids, size=valid_size, replace=False)
            self.valid_ids = np.array([id_ for id_ in self.patient_ids if id_ in valid_ids])
            self.test_ids = np.array([id_ for id_ in available_ids if id_ not in self.valid_ids])

        return self.train_ids, self.valid_ids, self.test_ids

    def transform_to_numeric_adanet(self):
        for name in self.df_names:
            df = getattr(self, name)
            df_processed = df.copy()
            date_cols_to_process = list(df.select_dtypes(include=['M8[ns]']).columns)
            for date_col in date_cols_to_process:
                # convert into days between 01.01.2022 and date
                df_processed[date_col] = (pd.to_datetime("01/05/2022") - df_processed[date_col]).dt.days
            setattr(self, name + '_proc', df_processed)
        # specific one hot encoding
        # self.a_visit_df_proc = pd.get_dummies(self.a_visit_df_proc, columns=[
        #                                      '.smoker', '.morning_stiffness_duration_radai'], drop_first=True)
        self.socio_df_proc = pd.get_dummies(self.socio_df_proc, columns=['smoker'])
        self.radai_df_proc = pd.get_dummies(self.radai_df_proc, columns=['morning_stiffness_duration_radai'])
        self.med_df_proc = pd.get_dummies(self.med_df_proc, columns=['medication_generic_drug', 'medication_drug_classification',
                                                                     ])
        self.patients_df_proc = pd.get_dummies(self.patients_df_proc, columns=[
                                               'gender', 'anti_ccp', 'ra_crit_rheumatoid_factor'])
        # transform to numeric
        columns_to_exclude = ['patient_id', 'uid_num', 'med_id']
        for name in self.df_names:
            df = getattr(self, name + '_proc')
            columns = [col for col in df.columns if col not in columns_to_exclude]
            df[columns] = df[columns].apply(pd.to_numeric, axis=1)
            setattr(self, name + '_proc', df)
        # TODO add condition here
        if hasattr(self, 'joint_df_proc'):
            self.joint_df_proc = pd.get_dummies(self.joint_df_proc, columns=[
                'smoker', 'morning_stiffness_duration_radai', 'medication_generic_drug', 'medication_drug_classification'])
        return

    def transform_to_numeric(self):
        mappings = {}
        for name in self.df_names:
            mappings[name] = {}
            df = getattr(self, name)
            df_processed = df.copy()
            date_cols_to_process = list(df.select_dtypes(include=['M8[ns]']).columns)
            for date_col in date_cols_to_process:
                # convert into days between 05.01.2022 and date
                # TODO change and import global variable instead
                df_processed[date_col] = (pd.to_datetime("01/05/2022") - df_processed[date_col]).dt.days
            
            # one hot encoding
            df_processed = get_dummies(df_processed)
            # transform categories
            if name == 'med_df':
                for col in ['medication_drug', 'medication_generic_drug']:
                    if col in df_processed.columns:
                        df_processed[col], mappings['med_df'][col] = map_category(
                            df_processed, col)
            if name == 'healthissues_df':
                for col in ['health_issue_category_1', 'health_issue_category_2']:
                    if col in df_processed.columns:
                        df_processed[col], mappings['healthissues_df'][col] = map_category(df_processed, col)
            if name == 'sonarra_df':
                if col in df_processed.columns:
                    for col in ['cartilage_body_entire_metacarp__al_joint_of_little_finger_right', 'cartilage_body_entire_metacarp__eal_joint_of_index_finger_right',
                                'cartilage_body_entire_metacarp__eal_joint_of_little_finger_left', 'cartilage_body_entire_metacarp__geal_joint_of_index_finger_left']:
                        df_processed[col], mappings['sonarra_df'][col] = map_category(df_processed, col)
            setattr(self, name + '_proc', df_processed)
        # specific one hot encoding
        # transform to numeric
        columns_to_exclude = ['patient_id', 'uid_num', 'med_id']
        for name in self.df_names:
            df = getattr(self, name + '_proc')
            columns = [col for col in df.columns if col not in columns_to_exclude]
            df[columns] = df[columns].apply(pd.to_numeric, axis=1)
        return

    def scale_and_tensor(self, nan_dummies=True):
        # (x-min)/(max-min)
        # attribute to keep track of all tensor names¨
        self.tensor_names = []
        self.tensor_indices_mapping_train = {index: {name: [] for name in self.df_names} for index in self.train_ids}
        self.tensor_indices_mapping_valid = {index: {name: [] for name in self.df_names} for index in self.valid_ids}
        self.tensor_indices_mapping_test = {index: {name: [] for name in self.df_names} for index in self.test_ids}
        for name in self.df_names:
            df = getattr(self, name + '_proc')

            # get train min and max values
            columns_to_exclude = ['patient_id', 'uid_num', 'med_id']
            columns = [col for col in df.columns if col not in columns_to_exclude]
            min_train_values = df[df.patient_id.isin(self.train_ids)][columns].min()
            max_train_values = df[df.patient_id.isin(self.train_ids)][columns].max()
            # to not scale classification targets

            if self.target_category_name in min_train_values.index:
                min_train_values[self.target_category_name] = 0
                max_train_values[self.target_category_name] = 1
            # store scaling values
            setattr(self, str(name) + '_scaling_values', (min_train_values, max_train_values))
            # scale everything
            df[columns] = (df[columns] - min_train_values) / (max_train_values - min_train_values)
            # change that, split into train, test, valid tensors and save mappings
            if nan_dummies:
                setattr(self, str(name) + '_scaled_tensor_train',
                        torch.tensor(df[df.patient_id.isin(self.train_ids)][columns].fillna(-1).values, dtype=torch.float32))
                setattr(self, str(name) + '_scaled_tensor_valid',
                        torch.tensor(df[df.patient_id.isin(self.valid_ids)][columns].fillna(-1).values, dtype=torch.float32))
                setattr(self, str(name) + '_scaled_tensor_test',
                        torch.tensor(df[df.patient_id.isin(self.test_ids)][columns].fillna(-1).values, dtype=torch.float32))
                self.tensor_names.extend([str(name) + '_scaled_tensor_train', str(name) +
                                         '_scaled_tensor_valid', str(name) + '_scaled_tensor_test'])
            else:
                setattr(self, str(name) + '_scaled_tensor_train',
                        torch.tensor(df[df.patient_id.isin(self.train_ids)][columns].values, dtype=torch.float32))
                setattr(self, str(name) + '_scaled_tensor_valid',
                        torch.tensor(df[df.patient_id.isin(self.valid_ids)][columns].values, dtype=torch.float32))
                setattr(self, str(name) + '_scaled_tensor_test',
                        torch.tensor(df[df.patient_id.isin(self.test_ids)][columns].values, dtype=torch.float32))
                self.tensor_names.extend([str(name) + '_scaled_tensor_train', str(name) +
                                         '_scaled_tensor_valid', str(name) + '_scaled_tensor_test'])
            # store mapping btw patient_ids and tensor indices
            df['tensor_indices_train'] = np.nan
            df['tensor_indices_valid'] = np.nan
            df['tensor_indices_test'] = np.nan
            df.loc[df.patient_id.isin(self.train_ids), 'tensor_indices_train'] = [index for index in range(
                len(df.loc[df.patient_id.isin(self.train_ids), 'tensor_indices_train']))]
            df.loc[df.patient_id.isin(self.valid_ids), 'tensor_indices_valid'] = [index for index in range(
                len(df.loc[df.patient_id.isin(self.valid_ids), 'tensor_indices_valid']))]
            df.loc[df.patient_id.isin(self.test_ids), 'tensor_indices_test'] = [index for index in range(
                len(df.loc[df.patient_id.isin(self.test_ids), 'tensor_indices_test']))]
            # indices mapping in tensors
            for index in self.train_ids:
                self.tensor_indices_mapping_train[index][name] = df[df.patient_id ==
                                                                    index]['tensor_indices_train'].values
            for index in self.test_ids:
                self.tensor_indices_mapping_test[index][name] = df[df.patient_id ==
                                                                   index]['tensor_indices_test'].values
            for index in self.valid_ids:
                self.tensor_indices_mapping_valid[index][name] = df[df.patient_id ==
                                                                    index]['tensor_indices_valid'].values
            # To retrieve scaled values directly from patients
            tensor_name = name + '_tensor'
            for patient in self.train_ids:
                value = getattr(self, str(name) +
                                '_scaled_tensor_train')[df[df.patient_id == patient]['tensor_indices_train'].values]
                self[patient].add_info(tensor_name, value)
            for patient in self.valid_ids:
                value = getattr(self, str(name) +
                                '_scaled_tensor_valid')[df[df.patient_id == patient]['tensor_indices_valid'].values]
                self[patient].add_info(tensor_name, value)
            for patient in self.test_ids:
                value = getattr(self, str(name) +
                                '_scaled_tensor_test')[df[df.patient_id == patient]['tensor_indices_test'].values]
                self[patient].add_info(tensor_name, value)
            # store column number of target and time to target visit
            if name == 'targets_df':
                self.target_index = list(df[columns].columns).index(self.target_category_name)
                self.time_index = list(df[columns].columns).index('date')
                self.target_value_index = list(df[columns].columns).index('das283bsr_score')

        # TODO have only one tensor and mapping to train, valid test (instead of 3 different ?)
        return

    def visit_count(self):
        # number of available visits per patient
        max_number_visits = max([len(pat.visits) for _, pat in self.patients.items()])
        self.visit_dict = {value: [index for index, patient in self.patients.items() if len(
            patient.visits) == value] for value in range(max_number_visits)}
        return
