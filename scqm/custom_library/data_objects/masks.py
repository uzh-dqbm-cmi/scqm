from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scqm.custom_library.data_objects.dataset import Dataset


class Masks:
    """Masks to handle patient timelines corresponding to each event"""

    def __init__(self, device: str, indices: list):
        """Instantiate

        Args:
            device (str): CPU or GPU
            indices (list): list of patient indices
        """
        self.device = device
        self.indices = indices

    def get_masks(
        self,
        dataset: Dataset,
        debug_patient: str,
        min_time_since_last_event: int = 15,
        max_time_since_last_event: int = 450,
    ):
        """Create masks for each event.

        Create for each event type (e.g. medication), for each patient a boolean tensor of the size of the patient timeline. The tensor is true if the event in the timeline is of the same type and False otherwise.

        Args:
            dataset (Dataset): dataset
            debug_patient (str): id of patient used for debugging
            min_time_since_last_event (int, optional): Minimum time until target visit (in days) to keep the event. Defaults to 30.
            max_time_since_last_event (int, optional): Maximum time (in days) since last event to keep visit as target. Defaults to 450.
        """
        # get max number of visits for a patient in subset
        self.min_time_since_last_event = min_time_since_last_event
        self.max_time_since_last_event = max_time_since_last_event
        self.num_targets = [
            len(dataset.patients[index].targets) for index in self.indices
        ]
        max_num_targets = max(self.num_targets)
        seq_lengths = torch.zeros(
            size=(
                max_num_targets - dataset.min_num_targets + 1,
                len(self.indices),
                len(dataset.event_names),
            ),
            dtype=torch.long,
            device=self.device,
        )
        # to store for each patient for each visit the visit/medication mask up to that visit. This mask allows
        # us to then easily combine the visit and medication events in the right order. True is for visit events and False for medications.
        # E.g. if a patient has the timeline [m1, m2, v1, m3, m4, v2, m5, v3] the corresponding masks up to each of the 3 visits would be
        # [[False, False], [False, False, True, False, False], [False, False, True, False, False, True, False]] and the sequence lengths
        # for visits/medication count up to each visit [[0, 2], [1, 4], [2, 5]]
        masks_dict = {
            event: [[] for i in range(len(self.indices))]
            for event in dataset.event_names
        }
        self.available_target_mask = torch.full(
            size=(len(self.indices), max_num_targets - dataset.min_num_targets + 1),
            fill_value=False,
            device=self.device,
        )
        self.target_category = torch.full(
            size=(len(self.indices), max_num_targets - dataset.min_num_targets + 1),
            fill_value=np.nan,
            device=self.device,
        )
        for i, patient in enumerate(self.indices):
            for target in range(
                0, len(dataset.patients[patient].targets) - dataset.min_num_targets + 1
            ):
                # get timeline up to visit (not included)
                (
                    seq_lengths[target, i, :],
                    cropped_timeline,
                    cropped_timeline_mask,
                    visual,
                    to_predict,
                    increase,
                ) = dataset.patients[patient].get_cropped_timeline(
                    target + dataset.min_num_targets,
                    min_time_since_last_event=min_time_since_last_event,
                    max_time_since_last_event=max_time_since_last_event,
                )
                self.available_target_mask[i, target] = to_predict
                self.target_category[i, target] = increase
                for event in dataset.event_names:
                    # masks_dict[event][i].append(torch.broadcast_to(torch.tensor([[True if tuple_[0] == event else False] for tuple_ in cropped_timeline_mask]),
                    #                                               (len(cropped_timeline_mask), model.size_embedding)))
                    masks_dict[event][i].append(
                        torch.tensor(
                            [
                                [True if tuple_[0] == event else False]
                                for tuple_ in cropped_timeline_mask
                            ]
                        ),
                    )

                if debug_patient and patient == debug_patient:
                    print(
                        f'target {target} cropped timeline mask {visual} visit mask {masks_dict["a_visit"][i]} medication mask {masks_dict["med"][i]}'
                    )

        # tensor of shape batch_size x max_num_targets with True in position (p, v) if patient p has at least v visits
        # and False else. we use this mask later to select the patients up to each visit.
        # self.available_target_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
        #                                           else False for index in range(dataset.min_num_targets, max_num_targets + 1)] for patient in self.indices], device=self.device)

        # stores for each patient in batch the total number of visits and medications
        # it is used later to index correctly the visits and medications dataframes
        # total num visits and meds

        self.total_num = torch.tensor(
            [
                [
                    getattr(dataset.patients[patient], "num_" + event + "_events")
                    for event in dataset.event_names
                ]
                for patient in self.indices
            ],
            device=self.device,
        )
        self.seq_lengths = seq_lengths

        for event in dataset.event_names:
            setattr(self, event + "_masks", masks_dict[event])

        # just for prints
        lengths = torch.zeros(len(dataset.event_names), device=self.device)
        number_hist = np.count_nonzero(np.array(self.available_target_mask.cpu()))
        for visit in range(seq_lengths.shape[0]):
            for event in range(seq_lengths.shape[2]):
                lengths[event] += seq_lengths[
                    visit, self.available_target_mask[:, visit], event
                ].sum()
        print(f"total num of histories {number_hist}")
        for event in range(seq_lengths.shape[2]):
            print(
                f"average number of events {dataset.event_names[event]} {(lengths[event])/number_hist}"
            )
        return
