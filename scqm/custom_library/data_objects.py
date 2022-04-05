import datetime
import numpy as np
import pandas as pd
import torch

class DataObject:
    def __init__(self, df_dict, patient_id):
        for name in df_dict.keys():
            setattr(self, str(name) + '_df', df_dict[name][df_dict[name]['patient_id'] == patient_id])
        return
    def add_info(self, name, value):
        setattr(self, name, value)
        return


class Patient(DataObject):
    #TODO be coherent with sorting (also other dfs)
    def __init__(self, df_dict, patient_id, event_names):
        super().__init__(df_dict, patient_id)
        self.id = patient_id
        self.visits = self.get_visits()
        self.medications = self.get_medications()
        self.event_names = event_names
        self.other_events = self.get_other_events()
        self.timeline = self.get_timeline()
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
        #total number of medication events counting all the available start and end dates
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
            setattr(self, 'num_'+name+'_events', len(df))
            for index in df.index:
                #TODO change maybe uid_num and put generic event id
                other_events.append(Event(name, df.loc[index, 'uid_num'], df.loc[index, 'date']))
        return other_events

    def get_timeline(self):
        #get sorted dates of events
        visit_event_list = [(self.visit_dates[index], 'a_visit', self.visit_ids[index]) for index in range(len(self.visit_dates))]
        med_sart_list = [(self.med_intervals[index][0], 'med', self.med_ids[index]) for index in range(len(self.med_intervals))]
        med_end_list = [(self.med_intervals[index][1], 'med', self.med_ids[index]) for index in range(len(self.med_intervals))]
        other_events = [(event.date, event.name, event.id) for event in self.other_events]
        all_events = visit_event_list + med_sart_list + med_end_list + other_events
        # the 'a', 'b', 'c' flags before visit and meds are there to ensure that if they occur at the same date, the visit comes before
        all_events.sort()
        #remove NaT events
        all_events = [event for event in all_events if not pd.isnull(event[0])]
        #get sorted ids of events, format : True --> visit, False --> medication along with id
        #e.g. [(True, visit_id), (False, med_id), ...]
        self.timeline_mask = [(event[1], event[2])
                              for event in all_events]
        self.timeline_visual = ['v' if event[1] == 'a_visit' else 'm' if event[1] == 'med' else event[1]
                                for event in all_events]
        self.mask = [[event[0]] for event in self.timeline_mask]
        return all_events
    
    def time_between_events(self, event='visit'):
        if event == 'visit':
            if len(self.visit_dates)==1:
                #TODO change that
                return 'only one visit'
            else:
                return ((self.visit_dates[-1] - self.visit_dates[0]).astype('timedelta64[D]') / np.timedelta64(1, 'D'))/(len(self.visit_dates)-1)
        else:
            #mean length of medication, mean duration between two starts of medications
            mean_length=0
            num_meds=0
            for med in self.medications:
                if not pd.isnull(med.interval[1]):
                    mean_length += (med.interval[1] - med.interval[0]) / np.timedelta64(1, 'D')
                    num_meds +=1
            if num_meds==0:
                mean_length = 'not_applicable'
            else:
                mean_length=mean_length/num_meds
            if len(self.medications) == 1:
                duration= 'not_applicable'
            else:
                duration = 0
                for index, med in enumerate(self.medications[1:]):
                    duration += (med.start_date - self.medications[index - 1].start_date)/ np.timedelta64(1, 'D')
                duration = duration / (len(self.medications) - 1)
            return mean_length, duration

    # retrieve timeline up to n-th visit
    def get_cropped_timeline(self, n=1):
        if n >  len(self.visits):
            raise ValueError('n bigger than number of visits')
            #return np.nan, np.nan, [], []
        # get all events up to n-th visit
        else :
            cropped_timeline = []
            num_of_each_event = torch.zeros(size=(len(self.event_names),))
            index_of_visit = self.event_names.index('a_visit')
            index = 0
            while (num_of_each_event[index_of_visit] < n and index < len(self.timeline)):
                event = self.timeline[index]
                cropped_timeline.append(event)
                index_of_event = self.event_names.index(event[1])
                num_of_each_event[index_of_event] += 1
                index += 1
            #remove last visit
            cropped_timeline = cropped_timeline[:-1]
            num_of_each_event[index_of_visit] -= 1
            cropped_timeline_mask = [(event[1], event[2])
                                  for event in cropped_timeline]
            cropped_timeline_visual = ['v' if event[1] == 'a_visit' else 'm' if event[1] == 'med' else 'e'
                                       for event in cropped_timeline]
            return num_of_each_event, cropped_timeline, cropped_timeline_mask, cropped_timeline_visual
  
class Event:
    def __init__(self, name, id, date =None):
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
        #start and end available
        if len(self.data) == 2: 
            self.start_date = self.data[self.data['is_start']==1]['date'].item()
            self.end_date = self.data[self.data['is_start']==0]['date'].item()
        # only start available
        else:
            self.start_date = self.data['date'].item()
            self.end_date=pd.NaT
        self.interval = [self.start_date, self.end_date]
        #self.get_related_visits()
        #TODO implement flags for missing data
        self.flag = None
        return

    def get_related_visits(self, tol=0):
        self.visit_start = []
        self.visit_end = []
        self.visit_during = []
        for visit in self.patient.visits:    
            #self.is_in_interval(visit) 
            self.is_close_to_interval(visit, tol)
        return
    
    
    def is_close_to_interval(self, visit_class, tol):

        tol = datetime.timedelta(days=tol)
        if self.interval[0] -tol <= visit_class.date <= self.interval[0]:
            self.visit_start.append(visit_class.id)
            #TODO change this dependency
            visit_class.med_start.append(self.id)
        elif self.interval[1] -tol <= visit_class.date <= self.interval[1]:
            self.visit_end.append(visit_class.id)
            visit_class.med_end.append(self.id)
        elif self.interval[0] < visit_class.date < self.interval[1] -tol:
            self.visit_during.append(visit_class.id)
            visit_class.med_during.append(self.id)
        # all of the above conditions are False if at least one of the dates is NaT
        return

#mapping and get_item
class Dataset:
    def __init__(self, df_dict, ids, target_name, event_names):
        self.initial_df_dict = df_dict
        self.patient_ids = list(ids)
        self.target_name = target_name
        self.event_names = event_names
        self.instantiate_patients()
        self.df_names = []
        for name in df_dict.keys():
            name_df = str(name) + '_df'
            setattr(self, name_df, df_dict[name][df_dict[name]['patient_id'].isin(ids)])
            self.df_names.append(name_df)
        return
    
    def instantiate_patients(self):
        self.patients = {id_: Patient(self.initial_df_dict, id_, self.event_names) for id_ in self.patient_ids}
        return 

    def drop(self, ids):
        #drop specific indices from dataset
        for id in ids:
            del self.patients[id]
            self.patient_ids.remove(id)
        return

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, x):
        return self.patients[x]
    
    def move_to_device(self, device):
        for tensor_name in self.tensor_names:
            setattr(self, tensor_name, getattr(self, tensor_name).to(device))
        return
    
    def split_data(self, prop_valid= 0.1, prop_test = 0.1):
        test_size = int(len(self)*prop_test)
        valid_size = int(len(self)*prop_valid)
        train_size = len(self) - valid_size - test_size
        train_ids = np.random.choice(self.patient_ids, size=train_size, replace=False)
        # order
        self.train_ids = np.array([id_ for id_ in self.patient_ids if id_ in train_ids])
        available_ids = [id_ for id_ in self.patient_ids if id_ not in self.train_ids]
        valid_ids = np.random.choice(available_ids, size = valid_size, replace=False)
        self.valid_ids = np.array([id_ for id_ in self.patient_ids if id_ in valid_ids])
        self.test_ids = np.array([id_ for id_ in available_ids if id_ not in self.valid_ids])
        return
    
    def transform_to_numeric(self):
        # change string categorical features and datetime objects
        feature_mapping = {}
        for name in self.df_names:
            feature_mapping[name] = {}
            df = getattr(self, name)
            df_processed = df.copy()
            string_cols_to_process = list(df.select_dtypes(include=[object]).columns)
            string_cols_to_process.remove('patient_id')
            date_cols_to_process = list(df.select_dtypes(include = ['M8[ns]']).columns)
            for col in string_cols_to_process:
                unique_values = df[col].dropna().unique()
                mapping = {elem : index for index, elem in enumerate(unique_values)}
                df_processed[col] = df_processed[col].replace(mapping)
                feature_mapping[name][col] = mapping
            for date_col in date_cols_to_process:
                # convert into days between 01.01.2022 and date
                df_processed[date_col] = (pd.to_datetime("01/01/2022")-df_processed[date_col]).dt.days
            setattr(self, name + '_proc', df_processed)
        print(f'{feature_mapping}')
        self.categorical_feature_mapping = feature_mapping
    
    def transform_to_numeric_adanet(self):
        for name in self.df_names:
            df = getattr(self, name)
            df_processed = df.copy()
            date_cols_to_process = list(df.select_dtypes(include=['M8[ns]']).columns)
            for date_col in date_cols_to_process:
                # convert into days between 01.01.2022 and date
                df_processed[date_col] = (pd.to_datetime("01/01/2022") - df_processed[date_col]).dt.days
            setattr(self, name + '_proc', df_processed)
        # specific one hot encoding
        # self.a_visit_df_proc = pd.get_dummies(self.a_visit_df_proc, columns=[
        #                                      '.smoker', '.morning_stiffness_duration_radai'], drop_first=True)
        self.socio_df_proc = pd.get_dummies(self.socio_df_proc, columns = ['.smoker'], drop_first=True)
        self.radai_df_proc = pd.get_dummies(self.radai_df_proc, columns=['.morning_stiffness_duration_radai'], drop_first=True)
        self.med_df_proc = pd.get_dummies(self.med_df_proc, columns=['medication_generic_drug', 'medication_drug_classification',
                                                                                   ], drop_first=True)
        self.patients_df_proc = pd.get_dummies(self.patients_df_proc, columns = ['gender', 'anti_ccp', 'ra_crit_rheumatoid_factor'], drop_first=True)
        #TODO add condition here
        if hasattr(self, 'joint_df_proc'):
            self.joint_df_proc = pd.get_dummies(self.joint_df_proc, columns=[
                '.smoker', '.morning_stiffness_duration_radai', 'medication_generic_drug', 'medication_drug_classification'], drop_first=True)
        return
    

    
    def scale_and_tensor(self, nan_dummies =True):
        #(x-min)/(max-min)
        # attribute to keep track of all tensor names¨
        self.tensor_names = []
        for name in self.df_names:
            df = getattr(self, name + '_proc')
            # get train min and max values
            columns_to_exclude = ['patient_id', 'uid_num', 'med_id']
            columns = [col for col in df.columns if col not in columns_to_exclude]
            min_train_values = df[df.patient_id.isin(self.train_ids)][columns].min()
            max_train_values = df[df.patient_id.isin(self.train_ids)][columns].max()

            # store scaling values
            setattr(self, str(name) + '_scaling_values', (min_train_values, max_train_values))
            # scale everything
            df[columns] = (df[columns]-min_train_values)/(max_train_values-min_train_values)
            #change that, split into train, test, valid tensors and save mappings
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

            # To retrieve scaled values directly from patients
            tensor_name = name + '_tensor'
            for patient in self.train_ids :
                value = getattr(self, str(name) + '_scaled_tensor_train')[df[df.patient_id == patient]['tensor_indices_train'].values]
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
                self.target_index = list(df[columns].columns).index(self.target_name)
                self.time_index = list(df[columns].columns).index('date')
                self.target_value_index = list(df[columns].columns).index('das283bsr_score')
        #TODO have only one tensor and mapping to train, valid test (instead of 3 different ?)
        return
    
    
    def visit_count(self):
        # number of available visits per patient
        max_number_visits = max([len(pat.visits) for _, pat in self.patients.items()])
        self.visit_dict = {value: [index for index, patient in self.patients.items() if len(
            patient.visits) == value] for value in range(max_number_visits)}
        return
    
