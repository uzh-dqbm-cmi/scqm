import numpy as np
from pandas import array

from scqm.custom_library.data_objects.dataset import Dataset


class Batch:
    """Base batch class for adaptivenet"""

    def __init__(
        self,
        device: str,
        all_indices: array,
        available_indices: array,
        current_indices=None,
        tensor_names: list = [],
        target_name: str = "",
        special_indices: list = [],
    ):
        """Instanciate object

        Args:
            device (str): device
            all_indices (array): all patient indices to consider
            available_indices (array): available indices for epoch
            current_indices (_type_, optional): currrent batch indices. Defaults to None.
        """
        self.device = device
        self.all_indices = all_indices
        self.available_indices = available_indices
        self.current_indices = current_indices
        self.tensor_names = tensor_names
        self.target_name = target_name
        self.special_indices = special_indices

    def get_batch(
        self,
        dataset: Dataset,
        batch_size=None,
        debug_patient=None,
        indices_to_include=[],
        indices_to_exclude=[],
    ):
        """
        Select batch of patients from available indices

        Args:
            dataset (Dataset): dataset
            batch_size (_type_, optional): Number of patients per batch. Defaults to None.
            debug_patient (_type_, optional): ID of patient for debugging. Defaults to None.

        Raises:
            ValueError: if some patients in visits dont correspond to patients in targets (and vice versa)
        """
        indices_to_include_next = []
        # during training, only select subset of available indices
        if batch_size:
            # batch size
            size = min(
                len(set(self.available_indices).difference(indices_to_exclude)),
                batch_size,
            )
            # batch and corresponding tensor indices
            if len(indices_to_include) > 0:
                self.current_indices = indices_to_include + list(
                    np.random.choice(
                        [
                            index
                            for index in self.available_indices
                            if index not in indices_to_include
                        ],
                        size=max(size - len(indices_to_include), 0),
                        replace=False,
                    )
                )
            elif len(indices_to_exclude) > 0:
                self.current_indices = np.random.choice(
                    [
                        index
                        for index in self.available_indices
                        if index not in indices_to_exclude
                    ],
                    size=size,
                    replace=False,
                )
            else:
                self.current_indices = np.random.choice(
                    self.available_indices, size=size, replace=False
                )
                if (
                    len(set(self.special_indices).intersection(self.current_indices))
                    > 0
                ):
                    indices_to_include_next = [
                        index
                        for index in self.current_indices
                        if index in self.special_indices
                    ]

            # print(f'len available indices {len(self.available_indices)}')

        for name in self.tensor_names:
            indices = [
                dataset.tensor_indices_mapping_train[patient][name + "_df"]
                for patient in self.current_indices
            ]
            if len(indices) > 0:
                indices = np.concatenate(indices)
            setattr(self, "indices_" + name, indices)
        if debug_patient and debug_patient in self.current_indices:
            self.debug_index = list(self.current_indices).index(debug_patient)
            print(f"index in batch {self.debug_index}")
        else:
            self.debug_index = None
        # if (self.indices_a_visit != self.indices_targets).any():
        #     raise ValueError("index mismatch between visits and targets")

        return indices_to_include_next

    def get_masks(self, dataset: Dataset, debug_patient):
        """Get event masks for patients in batch

        Args:
            dataset (Dataset): dataset
            debug_patient (_type_): ID of patient for debugging
        """
        if hasattr(dataset, "mapping_for_masks"):
            mapping = dataset.mapping_for_masks
            masks = dataset.masks
        else:
            if self.target_name == "das283bsr_score":

                mapping = dataset.mapping_for_masks_das28
                masks = dataset.masks_das28
            elif self.target_name == "asdas_score":
                mapping = dataset.mapping_for_masks_asdas
                masks = dataset.masks_asdas
        indices_mapping = [mapping[id_] for id_ in self.current_indices]
        self.seq_lengths = masks.seq_lengths[:, indices_mapping, :]
        for event in dataset.event_names:
            name = event + "_masks"
            setattr(
                self,
                name,
                list(getattr(masks, name)[i] for i in indices_mapping),
            )
        self.available_target_mask = masks.available_target_mask[indices_mapping]
        self.target_categories = masks.target_category[indices_mapping]
        self.total_num = masks.total_num[indices_mapping]
        if len(self.current_indices) > 0:
            self.max_num_targets = max(
                list(masks.num_targets[i] for i in indices_mapping)
            )
        else:
            self.max_num_targets = 0

        if debug_patient and debug_patient in self.current_indices:
            index = mapping[debug_patient]
            for visit in range(masks.num_targets[index] - dataset.min_num_targets + 1):
                _, _, _, visual, _ = dataset.patients[
                    debug_patient
                ].get_cropped_timeline(visit + dataset.min_num_targets)
                print(f"visit {visit} cropped timeline mask {visual}")

        return
