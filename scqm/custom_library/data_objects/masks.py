import torch
import numpy as np


class Masks:
    def __init__(self, device, indices):
        self.device = device
        self.indices = indices

    def get_masks(
        self,
        dataset,
        debug_patient,
        min_time_since_last_event=30,
        max_time_since_last_event=450,
    ):
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
        # get max number of visits for a patient in subset
        self.min_time_since_last_event = min_time_since_last_event
        self.max_time_since_last_event = max_time_since_last_event
        self.num_visits = [
            len(dataset.patients[index].visits) for index in self.indices
        ]
        max_num_visits = max(self.num_visits)
        seq_lengths = torch.zeros(
            size=(
                max_num_visits - dataset.min_num_visits + 1,
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
        self.available_visit_mask = torch.full(
            size=(len(self.indices), max_num_visits - dataset.min_num_visits + 1),
            fill_value=False,
            device=self.device,
        )
        for i, patient in enumerate(self.indices):
            for visit in range(
                0, len(dataset.patients[patient].visits) - dataset.min_num_visits + 1
            ):
                # get timeline up to visit (not included)
                (
                    seq_lengths[visit, i, :],
                    cropped_timeline,
                    cropped_timeline_mask,
                    visual,
                    to_predict,
                ) = dataset.patients[patient].get_cropped_timeline(
                    visit + dataset.min_num_visits,
                    min_time_since_last_event=min_time_since_last_event,
                    max_time_since_last_event=max_time_since_last_event,
                )
                self.available_visit_mask[i, visit] = to_predict
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
                        f'visit {visit} cropped timeline mask {visual} visit mask {masks_dict["a_visit"][i]} medication mask {masks_dict["med"][i]}'
                    )

        # tensor of shape batch_size x max_num_visits with True in position (p, v) if patient p has at least v visits
        # and False else. we use this mask later to select the patients up to each visit.
        # self.available_visit_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
        #                                           else False for index in range(dataset.min_num_visits, max_num_visits + 1)] for patient in self.indices], device=self.device)

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
        number_hist = np.count_nonzero(np.array(self.available_visit_mask.cpu()))
        for visit in range(seq_lengths.shape[0]):
            for event in range(seq_lengths.shape[2]):
                lengths[event] += seq_lengths[
                    visit, self.available_visit_mask[:, visit], event
                ].sum()
        print(f"total num of histories {number_hist}")
        for event in range(seq_lengths.shape[2]):
            print(
                f"average number of events {dataset.event_names[event]} {(lengths[event])/number_hist}"
            )
        return
