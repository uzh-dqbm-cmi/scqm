import numpy as np

class Batch:
    def __init__(self, device, all_indices, available_indices, current_indices=None):
        self.device = device
        self.all_indices = all_indices
        self.available_indices = available_indices
        self.current_indices = current_indices

    def get_batch(self, dataset, batch_size=None, debug_patient=None):
        """First, selects a batch of patients from the available indices for this epoch and the corresponding tensor (visits/medications/
        patient) slices. Then for each visit v, for each patient of the batch, create a mask to combine the visits/medication events coming before v in the right order.

        """
        # during training, only select subset of available indices
        if batch_size:
            # batch size
            size = min(len(self.available_indices), batch_size)
            # batch and corresponding tensor indices
            self.current_indices = np.random.choice(
                self.available_indices, size=size, replace=False
            )
            # print(f'len available indices {len(self.available_indices)}')

        for name in dataset.event_names + ["patients", "targets"]:
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
        if (self.indices_a_visit != self.indices_targets).any():
            raise ValueError("index mismatch between visits and targets")

        return

    def get_masks(self, dataset, debug_patient):
        """_summary_

        Args:
            dataset (_type_): _description_
            subset (_type_): _description_
            min_num_visits (_type_): min number of initial visits to retrieve the information from
        e.g. if min_num_visits = 2, for each patient we start retrieving all information
        up to the 2nd visit, i.e. medications before 2nd visit and info about 1st visit
        (in other words, min_num_visits is the first target visit). For each visit v >= min_num_visits, we store for each patient the number of visits and medication events
        up to v

        Returns:
            _type_: _description_
        """

        indices_mapping = [
            dataset.mapping_for_masks[id_] for id_ in self.current_indices
        ]
        self.seq_lengths = dataset.masks.seq_lengths[:, indices_mapping, :]
        for event in dataset.event_names:
            name = event + "_masks"
            setattr(
                self,
                name,
                list(getattr(dataset.masks, name)[i] for i in indices_mapping),
            )
        self.available_visit_mask = dataset.masks.available_visit_mask[indices_mapping]
        self.total_num = dataset.masks.total_num[indices_mapping]
        self.max_num_visits = max(
            list(dataset.masks.num_visits[i] for i in indices_mapping)
        )

        if debug_patient and debug_patient in self.current_indices:
            index = dataset.mapping_for_masks[debug_patient]
            for visit in range(
                dataset.masks.num_visits[index] - dataset.min_num_visits + 1
            ):
                _, _, _, visual, _ = dataset.patients[
                    debug_patient
                ].get_cropped_timeline(visit + dataset.min_num_visits)
                print(f"visit {visit} cropped timeline mask {visual}")

        return
