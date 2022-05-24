import numpy as np
import pandas as pd
import torch
import pickle

from tqdm import tqdm

from scqm.custom_library.data_objects.patient import Patient
from scqm.custom_library.data_objects.masks import Masks
from scqm.custom_library.preprocessing.utils import map_category, get_dummies
from scqm.custom_library.global_vars import *


class Dataset:
    def __init__(
        self,
        device,
        df_dict,
        ids,
        target_category_name,
        event_names,
        min_num_visits,
        mapping=None,
    ):
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
        print(f"Getting masks....")
        self.mapping_for_masks = {
            patient: index for index, patient in enumerate(self.patient_ids)
        }
        self.reverse_mapping_for_masks = {
            value: key for key, value in self.mapping_for_masks.items()
        }
        self.masks = Masks(self.device, self.patient_ids)
        self.masks.get_masks(
            self,
            debug_patient=None,
            min_time_since_last_event=min_time_since_last_event,
            max_time_since_last_event=max_time_since_last_event,
        )
        self.stratifier = {
            num_visit: [
                self.reverse_mapping_for_masks[patient_index]
                for patient_index in range(len(self.masks.num_visits))
                if self.masks.num_visits[patient_index] == num_visit
            ]
            for num_visit in range(1, max(self.masks.num_visits) + 1)
        }
        return

    def instantiate_patients(self):
        self.patients = {
            id_: Patient(self.initial_df_dict, id_, self.event_names)
            for id_ in tqdm(self.patient_ids)
        }

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
        with open(
            "/opt/data/processed/patients_satisfying_inclusion_criteria.pickle", "rb"
        ) as handle:
            patients_to_keep = pickle.load(handle)
        to_drop = list(set(self.patient_ids).difference(patients_to_keep))
        print(
            f"Dropping {len(to_drop)} patients because they dont satisfy inclusion criteria"
        )
        self.drop(to_drop)
        print(f"{len(self)} patients remaining")
        return

    def create_dfs(self):
        self.df_names = []
        for name in self.initial_df_dict.keys():
            name_df = str(name) + "_df"
            setattr(
                self,
                name_df,
                self.initial_df_dict[name][
                    self.initial_df_dict[name]["patient_id"].isin(self.patient_ids)
                ],
            )
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
                train_i = np.random.choice(
                    available_ids, size=train_size, replace=False
                )
                self.train_ids.extend(train_i)
                available_ids = [id_ for id_ in available_ids if id_ not in train_i]
                valid_i = np.random.choice(
                    available_ids, size=valid_size, replace=False
                )
                self.valid_ids.extend(valid_i)
                self.test_ids.extend(
                    [id_ for id_ in available_ids if id_ not in valid_i]
                )
            self.train_ids = np.array(self.train_ids)
            self.valid_ids = np.array(self.valid_ids)
            self.test_ids = np.array(self.test_ids)
        else:
            test_size = int(len(self) * prop_test)
            valid_size = int(len(self) * prop_valid)
            train_size = len(self) - valid_size - test_size
            train_ids = np.random.choice(
                self.patient_ids, size=train_size, replace=False
            )
            # order
            self.train_ids = np.array(
                [id_ for id_ in self.patient_ids if id_ in train_ids]
            )
            available_ids = [
                id_ for id_ in self.patient_ids if id_ not in self.train_ids
            ]
            valid_ids = np.random.choice(available_ids, size=valid_size, replace=False)
            self.valid_ids = np.array(
                [id_ for id_ in self.patient_ids if id_ in valid_ids]
            )
            self.test_ids = np.array(
                [id_ for id_ in available_ids if id_ not in self.valid_ids]
            )

        return self.train_ids, self.valid_ids, self.test_ids

    def transform_to_numeric_adanet(self, real_data=True):
        for name in self.df_names:
            df = getattr(self, name)
            df_processed = df.copy()
            date_cols_to_process = list(df.select_dtypes(include=["M8[ns]"]).columns)
            for date_col in date_cols_to_process:
                # convert into days between 01.01.2022 and date
                df_processed[date_col] = (
                    pd.to_datetime(REFERENCE_DATE) - df_processed[date_col]
                ).dt.days
            setattr(self, name + "_proc", df_processed)
        # specific one hot encoding
        # self.a_visit_df_proc = pd.get_dummies(self.a_visit_df_proc, columns=[
        #                                      '.smoker', '.morning_stiffness_duration_radai'], drop_first=True)
        if real_data:
            self.socio_df_proc = pd.get_dummies(self.socio_df_proc, columns=["smoker"])
            self.radai_df_proc = pd.get_dummies(
                self.radai_df_proc, columns=["morning_stiffness_duration_radai"]
            )
        self.med_df_proc = pd.get_dummies(
            self.med_df_proc,
            columns=[
                "medication_generic_drug",
                "medication_drug_classification",
            ],
        )
        self.patients_df_proc = pd.get_dummies(
            self.patients_df_proc,
            columns=["gender", "anti_ccp", "ra_crit_rheumatoid_factor"],
        )
        # transform to numeric
        columns_to_exclude = ["patient_id", "uid_num", "med_id"]
        for name in self.df_names:
            df = getattr(self, name + "_proc")
            columns = [col for col in df.columns if col not in columns_to_exclude]
            df[columns] = df[columns].apply(pd.to_numeric, axis=1)
            setattr(self, name + "_proc", df)
        # TODO add condition here
        if hasattr(self, "joint_df_proc"):
            self.joint_df_proc = pd.get_dummies(
                self.joint_df_proc,
                columns=[
                    "smoker",
                    "morning_stiffness_duration_radai",
                    "medication_generic_drug",
                    "medication_drug_classification",
                ],
            )
        return

    def transform_to_numeric(self):
        mappings = {}
        for name in self.df_names:
            mappings[name] = {}
            df = getattr(self, name)
            df_processed = df.copy()
            date_cols_to_process = list(df.select_dtypes(include=["M8[ns]"]).columns)
            for date_col in date_cols_to_process:
                # convert into days between 05.01.2022 and date
                # TODO change and import global variable instead
                df_processed[date_col] = (
                    pd.to_datetime(REFERENCE_DATE) - df_processed[date_col]
                ).dt.days

            # one hot encoding
            df_processed = get_dummies(df_processed)
            # transform categories
            if name == "med_df":
                for col in ["medication_drug", "medication_generic_drug"]:
                    if col in df_processed.columns:
                        df_processed[col], mappings["med_df"][col] = map_category(
                            df_processed, col
                        )
            if name == "healthissues_df":
                for col in ["health_issue_category_1", "health_issue_category_2"]:
                    if col in df_processed.columns:
                        (
                            df_processed[col],
                            mappings["healthissues_df"][col],
                        ) = map_category(df_processed, col)
            if name == "sonarra_df":
                if col in df_processed.columns:
                    for col in [
                        "cartilage_body_entire_metacarp__al_joint_of_little_finger_right",
                        "cartilage_body_entire_metacarp__eal_joint_of_index_finger_right",
                        "cartilage_body_entire_metacarp__eal_joint_of_little_finger_left",
                        "cartilage_body_entire_metacarp__geal_joint_of_index_finger_left",
                    ]:
                        df_processed[col], mappings["sonarra_df"][col] = map_category(
                            df_processed, col
                        )
            setattr(self, name + "_proc", df_processed)
        # specific one hot encoding
        # transform to numeric
        columns_to_exclude = ["patient_id", "uid_num", "med_id"]
        for name in self.df_names:
            df = getattr(self, name + "_proc")
            columns = [col for col in df.columns if col not in columns_to_exclude]
            df[columns] = df[columns].apply(pd.to_numeric, axis=1)
        return

    def scale_and_tensor(self, nan_dummies=True):
        # (x-min)/(max-min)
        # attribute to keep track of all tensor names¨
        self.tensor_names = []
        self.tensor_indices_mapping_train = {
            index: {name: [] for name in self.df_names} for index in self.train_ids
        }
        self.tensor_indices_mapping_valid = {
            index: {name: [] for name in self.df_names} for index in self.valid_ids
        }
        self.tensor_indices_mapping_test = {
            index: {name: [] for name in self.df_names} for index in self.test_ids
        }
        for name in self.df_names:
            df = getattr(self, name + "_proc")

            # get train min and max values
            columns_to_exclude = ["patient_id", "uid_num", "med_id"]
            columns = [col for col in df.columns if col not in columns_to_exclude]
            min_train_values = df[df.patient_id.isin(self.train_ids)][columns].min()
            max_train_values = df[df.patient_id.isin(self.train_ids)][columns].max()
            # to not scale classification targets

            if self.target_category_name in min_train_values.index:
                min_train_values[self.target_category_name] = 0
                max_train_values[self.target_category_name] = 1
            # store scaling values
            setattr(
                self,
                str(name) + "_scaling_values",
                (min_train_values, max_train_values),
            )
            # scale everything
            df[columns] = (df[columns] - min_train_values) / (
                max_train_values - min_train_values
            )
            # change that, split into train, test, valid tensors and save mappings
            if nan_dummies:
                setattr(
                    self,
                    str(name) + "_scaled_tensor_train",
                    torch.tensor(
                        df[df.patient_id.isin(self.train_ids)][columns]
                        .fillna(-1)
                        .values,
                        dtype=torch.float32,
                    ),
                )
                setattr(
                    self,
                    str(name) + "_scaled_tensor_valid",
                    torch.tensor(
                        df[df.patient_id.isin(self.valid_ids)][columns]
                        .fillna(-1)
                        .values,
                        dtype=torch.float32,
                    ),
                )
                setattr(
                    self,
                    str(name) + "_scaled_tensor_test",
                    torch.tensor(
                        df[df.patient_id.isin(self.test_ids)][columns]
                        .fillna(-1)
                        .values,
                        dtype=torch.float32,
                    ),
                )
                self.tensor_names.extend(
                    [
                        str(name) + "_scaled_tensor_train",
                        str(name) + "_scaled_tensor_valid",
                        str(name) + "_scaled_tensor_test",
                    ]
                )
            else:
                setattr(
                    self,
                    str(name) + "_scaled_tensor_train",
                    torch.tensor(
                        df[df.patient_id.isin(self.train_ids)][columns].values,
                        dtype=torch.float32,
                    ),
                )
                setattr(
                    self,
                    str(name) + "_scaled_tensor_valid",
                    torch.tensor(
                        df[df.patient_id.isin(self.valid_ids)][columns].values,
                        dtype=torch.float32,
                    ),
                )
                setattr(
                    self,
                    str(name) + "_scaled_tensor_test",
                    torch.tensor(
                        df[df.patient_id.isin(self.test_ids)][columns].values,
                        dtype=torch.float32,
                    ),
                )
                self.tensor_names.extend(
                    [
                        str(name) + "_scaled_tensor_train",
                        str(name) + "_scaled_tensor_valid",
                        str(name) + "_scaled_tensor_test",
                    ]
                )
            # store mapping btw patient_ids and tensor indices
            df["tensor_indices_train"] = np.nan
            df["tensor_indices_valid"] = np.nan
            df["tensor_indices_test"] = np.nan
            df.loc[df.patient_id.isin(self.train_ids), "tensor_indices_train"] = [
                index
                for index in range(
                    len(
                        df.loc[
                            df.patient_id.isin(self.train_ids), "tensor_indices_train"
                        ]
                    )
                )
            ]
            df.loc[df.patient_id.isin(self.valid_ids), "tensor_indices_valid"] = [
                index
                for index in range(
                    len(
                        df.loc[
                            df.patient_id.isin(self.valid_ids), "tensor_indices_valid"
                        ]
                    )
                )
            ]
            df.loc[df.patient_id.isin(self.test_ids), "tensor_indices_test"] = [
                index
                for index in range(
                    len(
                        df.loc[df.patient_id.isin(self.test_ids), "tensor_indices_test"]
                    )
                )
            ]
            # indices mapping in tensors
            for index in self.train_ids:
                self.tensor_indices_mapping_train[index][name] = df[
                    df.patient_id == index
                ]["tensor_indices_train"].values
            for index in self.test_ids:
                self.tensor_indices_mapping_test[index][name] = df[
                    df.patient_id == index
                ]["tensor_indices_test"].values
            for index in self.valid_ids:
                self.tensor_indices_mapping_valid[index][name] = df[
                    df.patient_id == index
                ]["tensor_indices_valid"].values
            # To retrieve scaled values directly from patients
            tensor_name = name + "_tensor"
            for patient in self.train_ids:
                value = getattr(self, str(name) + "_scaled_tensor_train")[
                    df[df.patient_id == patient]["tensor_indices_train"].values
                ]
                self[patient].add_info(tensor_name, value)
            for patient in self.valid_ids:
                value = getattr(self, str(name) + "_scaled_tensor_valid")[
                    df[df.patient_id == patient]["tensor_indices_valid"].values
                ]
                self[patient].add_info(tensor_name, value)
            for patient in self.test_ids:
                value = getattr(self, str(name) + "_scaled_tensor_test")[
                    df[df.patient_id == patient]["tensor_indices_test"].values
                ]
                self[patient].add_info(tensor_name, value)
            # store column number of target and time to target visit
            if name == "targets_df":
                self.target_index = list(df[columns].columns).index(
                    self.target_category_name
                )
                self.time_index = list(df[columns].columns).index("date")
                self.target_value_index = list(df[columns].columns).index(
                    "das283bsr_score"
                )

        # TODO have only one tensor and mapping to train, valid test (instead of 3 different ?)
        return

    def visit_count(self):
        # number of available visits per patient
        max_number_visits = max([len(pat.visits) for _, pat in self.patients.items()])
        self.visit_dict = {
            value: [
                index
                for index, patient in self.patients.items()
                if len(patient.visits) == value
            ]
            for value in range(max_number_visits)
        }
        return
