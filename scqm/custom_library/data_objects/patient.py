from scqm.custom_library.data_objects.data_object import DataObject
from scqm.custom_library.data_objects.visit import Visit
from scqm.custom_library.data_objects.medication import Medication
from scqm.custom_library.data_objects.event import Event
import torch
import pandas as pd


class Patient(DataObject):
    """Patient class"""

    def __init__(self, df_dict: dict, patient_id: str, event_names: list):
        """Instantiate a given patient

        Args:
            df_dict (dict): dictionnary of available dataframes
            patient_id (str): Unique id of patient
            event_names (list): possible events (e.g. visit, medication, ...)
        """
        super().__init__(df_dict, patient_id)
        self.id = patient_id
        self.visits = self.get_visits()
        self.medications = self.get_medications()
        self.event_names = event_names
        self.other_events = self.get_other_events()
        self.timeline = self.get_timeline()
        self.visits_to_predict = self.visits
        return

    def get_visits(self) -> list:
        """Create visit objects realted to that patient

        Returns:
            list: visit objects
        """
        self.visits_df = self.a_visit_df.sort_values(by="date", axis=0)
        self.visit_ids = self.visits_df["uid_num"].values
        self.visit_dates = self.visits_df["date"].values
        self.num_a_visit_events = len(self.visit_ids)
        visits = []
        for id_, date in zip(self.visit_ids, self.visit_dates):
            visits.append(Visit(self, id_, date))
        return visits

    def get_medications(self) -> list:
        """Create medication objects related to that patient

        Returns:
            list: medication objects
        """
        self.medications_df = self.med_df.sort_values(by="date", axis=0)
        self.med_ids = self.medications_df["med_id"].unique()
        # total number of medication events counting all the available start and end dates
        self.num_med_events = len(self.medications_df["med_id"])
        self.med_intervals = []
        meds = []
        for id_ in self.med_ids:
            m = Medication(self, id_)
            meds.append(m)
            self.med_intervals.append(m.interval)
        return meds

    def get_other_events(self) -> list:
        """Create other events related to that patient

        Returns:
            list: event objects
        """
        other_events = []
        events = [name for name in self.event_names if name not in ["a_visit", "med"]]
        for name in events:
            df = getattr(self, name + "_df")
            setattr(self, "num_" + name + "_events", len(df))
            for index in df.index:
                # TODO change maybe uid_num and put generic event id
                if "uid_num" in df.columns:
                    other_events.append(
                        Event(name, df.loc[index, "uid_num"], df.loc[index, "date"])
                    )
                else:
                    other_events.append(Event(name, date=df.loc[index, "date"]))
        return other_events

    def get_timeline(self) -> list:
        """Compute timeline of patient

        Returns:
            list: of ordered events
        """
        # get sorted dates of events
        visit_event_list = [
            (self.visit_dates[index], "a_visit", self.visit_ids[index])
            for index in range(len(self.visit_dates))
        ]
        med_sart_list = [
            (self.med_intervals[index][0], "med_s", self.med_ids[index])
            for index in range(len(self.med_intervals))
        ]
        med_end_list = [
            (self.med_intervals[index][1], "med_e", self.med_ids[index])
            for index in range(len(self.med_intervals))
        ]
        other_events = [
            (event.date, event.name, event.id) for event in self.other_events
        ]
        all_events = visit_event_list + med_sart_list + med_end_list + other_events
        # the 'a', 'b', 'c' flags before visit and meds are there to ensure that if they occur at the same date, the visit comes before
        # remove NaT events
        all_events = [event for event in all_events if not pd.isnull(event[0])]
        all_events.sort()
        # get sorted ids of events, format : True --> visit, False --> medication along with id
        # e.g. [(True, visit_id), (False, med_id), ...]
        self.timeline_mask = [
            (event[1], event[2])
            if event[1] not in ["med_s", "med_e"]
            else ("med", event[2])
            for event in all_events
        ]

        self.timeline_visual = [
            "v"
            if event[1] == "a_visit"
            else "m_s"
            if event[1] == "med_s"
            else "m_e"
            if event[1] == "m_e"
            else event[1]
            for event in all_events
        ]
        self.mask = [[event[0]] for event in self.timeline_mask]

        return all_events

    def get_cropped_timeline(
        self,
        n: int = 1,
        min_time_since_last_event: int = 30,
        max_time_since_last_event: int = 450,
    ):
        """Get cropped timeline up to a given visit.

        Args:
            n (int, optional): Number of visits up to which to retrieve timeline. Defaults to 1.
            min_time_since_last_event (int, optional): Minimum time (in days) to target visit to keep event. Defaults to 30.
            max_time_since_last_event (int, optional): Maximum time (in days) since last event to keep visit as target. Defaults to 450.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if n > len(self.visits):
            raise ValueError("n bigger than number of visits")
        # get all events up to n-th visit
        else:
            cropped_timeline = []
            num_of_each_event = torch.zeros(size=(len(self.event_names),))
            index_of_visit = self.event_names.index("a_visit")
            index = 0
            # date of n-th visit
            date_nth_visit = pd.Timestamp(self.visits[n - 1].date)
            # while number of visits < n and while ? other part is redundant no ?
            while (
                num_of_each_event[index_of_visit] < n
                and index < len(self.timeline)
                and (date_nth_visit - self.timeline[index][0]).days
                > min_time_since_last_event
            ):
                event = self.timeline[index]
                cropped_timeline.append(event)
                if event[1] in ["med_s", "med_e"]:
                    index_of_event = self.event_names.index("med")
                else:
                    index_of_event = self.event_names.index(event[1])
                num_of_each_event[index_of_event] += 1
                index += 1
            time_to_next = (date_nth_visit - self.timeline[index - 1][0]).days

            cropped_timeline_mask = [
                (event[1], event[2])
                if event[1] not in ["med_s", "med_e"]
                else ("med", event[2])
                for event in cropped_timeline
            ]
            cropped_timeline_visual = [
                "v"
                if event[1] == "a_visit"
                else "m_s"
                if event[1] == "med_s"
                else "m_e"
                if event[1] == "med_e"
                else "e"
                for event in cropped_timeline
            ]
            to_predict = True if len(cropped_timeline) > 0 else False
            if to_predict and (
                (date_nth_visit - cropped_timeline[-1][0]).days
                > max_time_since_last_event
            ):
                to_predict = False
            return (
                num_of_each_event,
                cropped_timeline,
                cropped_timeline_mask,
                cropped_timeline_visual,
                to_predict,
            )
