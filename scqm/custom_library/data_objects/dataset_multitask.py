from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.data_objects.patient import Patient
from scqm.custom_library.data_objects.masks import Masks
from tqdm import tqdm
from datetime import timedelta
import pandas as pd


class DatasetMultitask(Dataset):
    def __init__(
        self,
        device: str,
        df_dict: dict,
        ids: list,
        target_names: list,
        event_names: list,
        min_num_targets: int,
        mapping=None,
    ):
        """Instantiate object.

        Args:
            device (str): CPU or GPU
            df_dict (dict): Dictionnary of datframes
            ids (list): Patient ids to keep
            target_category_name (str): Name of categorical target
            event_names (list): Names of possible events (e.g. visit, medication)
            min_num_targets (int): Minimum number of visits to keep patient
            mapping (_type_, optional): Mapping used for categorical features. Defaults to None.
        """
        self.initial_df_dict = df_dict
        self.patient_ids = list(ids)
        self.target_names = target_names
        self.event_names = event_names
        self.min_num_targets = min_num_targets
        self.device = device
        self.instantiate_patients()
        self.target_category_name = "None"
        if mapping is not None:
            self.mapping = mapping

        return

    def instantiate_patients(self):
        """
        Instantiate all patient objects.
        """
        self.patients = {}
        self.multitarget_ids = []
        self.das28_ids = []
        self.asdas_ids = []
        for id_ in tqdm(self.patient_ids):
            p = Patient(
                self.initial_df_dict, id_, self.event_names, self.min_num_targets
            )
            if p.target_name != "None":
                self.patients[id_] = p
            if p.target_name == "both":
                self.multitarget_ids.append(id_)
            if p.target_name == "das283bsr_score":
                self.das28_ids.append(id_)
            if p.target_name == "asdas_score":
                self.asdas_ids.append(id_)
        print(
            f"Dropping {len(self.patient_ids)- len(self.patients)} because they have not enough temporality in the targets"
        )
        self.patient_ids = list(self.patients.keys())

    def get_masks(
        self, min_time_since_last_event: int = 15, max_time_since_last_event: int = 450
    ) -> None:
        """Get the event masks for each patient.

        For each event, for each element in the timeline the corresponding mask is true if the element is of type event else false.
        Args:
            min_time_since_last_event (int, optional): Minimum elapsed time (in days) to predicted visit to keep event. Defaults to 30.
            max_time_since_last_event (int, optional): Maximimum elapsed time (in days) to last event to keep visit as target. Defaults to 450.
        """
        print(f"Getting masks....")
        self.mapping_for_masks_das28 = {
            patient: index
            for index, patient in enumerate(self.das28_ids + self.multitarget_ids)
        }
        self.reverse_mapping_for_masks_das28 = {
            value: key for key, value in self.mapping_for_masks_das28.items()
        }
        self.masks_das28 = Masks(self.device, self.das28_ids + self.multitarget_ids)
        self.masks_das28.get_masks(
            self,
            debug_patient=None,
            min_time_since_last_event=min_time_since_last_event,
            max_time_since_last_event=max_time_since_last_event,
            target_name="das283bsr_score",
        )
        self.mapping_for_masks_asdas = {
            patient: index
            for index, patient in enumerate(self.asdas_ids + self.multitarget_ids)
        }
        self.reverse_mapping_for_masks_asdas = {
            value: key for key, value in self.mapping_for_masks_asdas.items()
        }
        self.masks_asdas = Masks(self.device, self.asdas_ids + self.multitarget_ids)
        self.masks_asdas.get_masks(
            self,
            debug_patient=None,
            min_time_since_last_event=min_time_since_last_event,
            max_time_since_last_event=max_time_since_last_event,
            target_name="asdas_score",
        )

        # self.mapping_for_masks_mult = {
        #     patient: index for index, patient in enumerate(self.multitarget_ids)
        # }
        # self.reverse_mapping_for_masks_mult = {
        #     value: key for key, value in self.mapping_for_masks_mult.items()
        # }
        # self.masks_mult = Masks(self.device, self.multitarget_ids)
        # self.masks_mult.get_masks(
        #     self,
        #     debug_patient=None,
        #     min_time_since_last_event=min_time_since_last_event,
        #     max_time_since_last_event=max_time_since_last_event,
        # )
        # stratifier on number of targets
        self.stratifier = {
            num_target: [
                self.reverse_mapping_for_masks_das28[patient_index]
                for patient_index in range(len(self.masks_das28.num_targets))
                if self.masks_das28.num_targets[patient_index] == num_target
            ]
            for num_target in range(1, max(self.masks_das28.num_targets) + 1)
        }
        for num_target in range(1, max(self.masks_asdas.num_targets) + 1):
            if num_target in self.stratifier.keys():
                self.stratifier[num_target].extend(
                    [
                        self.reverse_mapping_for_masks_asdas[patient_index]
                        for patient_index in range(len(self.masks_asdas.num_targets))
                        if self.masks_asdas.num_targets[patient_index] == num_target
                        and self.reverse_mapping_for_masks_asdas[patient_index]
                        not in self.multitarget_ids
                    ]
                )
            else:
                self.stratifier[num_target] = [
                    self.reverse_mapping_for_masks_asdas[patient_index]
                    for patient_index in range(len(self.masks_asdas.num_targets))
                    if self.masks_asdas.num_targets[patient_index] == num_target
                    and self.reverse_mapping_for_masks_asdas[patient_index]
                    not in self.multitarget_ids
                ]

        return

    def drop(self, ids: list):
        """Drop specific patients from dataset

        Args:
            ids (list): List of patient ids to drop
        """
        if not isinstance(ids, list):
            ids = list(ids)
        for id in ids:
            if id in self.patient_ids:
                del self.patients[id]
                self.patient_ids.remove(id)
                if id in self.basdai_ids:
                    self.basdai_ids.remove(id)
                if id in self.das28_ids:
                    self.das28_ids.remove(id)
                if id in self.multitarget_ids:
                    self.multitarget_ids.remove(id)
        return

    def move_to_device(self, device: str) -> None:
        """Move all tensors to device

        Args:
            device (str): CPU or GPU
        """
        for tensor_name in self.tensor_names:
            setattr(self, tensor_name, getattr(self, tensor_name).to(device))
        self.masks_das28.to_device(device, self.event_names)
        self.masks_asdas.to_device(device, self.event_names)
        return

    def post_process_joint_df(self, min_time_since_last_event: int = 15):
        joint_df = self.initial_df_dict["joint"]

        joint_df_das28 = pd.DataFrame(columns=list(joint_df.columns) + ["time_to_pred"])
        joint_targets_das28 = pd.DataFrame(columns=["patient_id", "value", "uid_num"])
        joint_df_asdas = pd.DataFrame(columns=list(joint_df.columns) + ["time_to_pred"])
        joint_targets_asdas = pd.DataFrame(columns=["patient_id", "value", "uid_num"])
        print("post-process joint df")
        for patient in tqdm(self.das28_ids + self.multitarget_ids):
            for target in self[patient].targets_to_predict["das283bsr_score"]:
                id_ = target.id
                date = target.date
                tmp = joint_df[joint_df.patient_id == patient]
                target_value = tmp[
                    (tmp.date == date) & (tmp.uid_num == id_)
                ].das283bsr_score.item()
                joint_targets_das28 = pd.concat(
                    [
                        joint_targets_das28,
                        pd.DataFrame(
                            {
                                "patient_id": [patient],
                                "value": [target_value],
                                "uid_num": [id_],
                            }
                        ),
                    ],
                    axis=0,
                )
                features = tmp[
                    (date - tmp.date) > timedelta(days=min_time_since_last_event)
                ].iloc[-1]
                features["time_to_pred"] = date
                joint_df_das28 = pd.concat((pd.DataFrame([features]), joint_df_das28))
        for patient in tqdm(self.asdas_ids + self.multitarget_ids):
            for target in self[patient].targets_to_predict["asdas_score"]:
                id_ = target.id
                date = target.date
                tmp = joint_df[joint_df.patient_id == patient]
                target_value = tmp[
                    (tmp.date == date) & (tmp.uid_num == id_)
                ].asdas_score.item()
                joint_targets_asdas = pd.concat(
                    [
                        joint_targets_asdas,
                        pd.DataFrame(
                            {
                                "patient_id": [patient],
                                "value": [target_value],
                                "uid_num": [id_],
                            }
                        ),
                    ],
                    axis=0,
                )
                features = tmp[
                    (date - tmp.date) > timedelta(days=min_time_since_last_event)
                ].iloc[-1]
                features["time_to_pred"] = date
                joint_df_asdas = pd.concat((pd.DataFrame([features]), joint_df_asdas))
        # columns to keep as features
        columns_to_drop = ["date"]
        self.initial_df_dict["joint_das28"] = joint_df_das28.drop(
            columns=columns_to_drop
        )
        self.initial_df_dict["joint_asdas"] = joint_df_asdas.drop(
            columns=columns_to_drop
        )
        self.initial_df_dict["joint_targets_das28"] = joint_targets_das28
        self.initial_df_dict["joint_targets_asdas"] = joint_targets_asdas

        return
