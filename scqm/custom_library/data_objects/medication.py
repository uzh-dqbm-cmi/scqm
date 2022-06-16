from __future__ import annotations
from scqm.custom_library.data_objects.event import Event
from scqm.custom_library.data_objects.patient import Visit

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scqm.custom_library.data_objects.patient import Patient
import datetime
import pandas as pd


class Medication(Event):
    """Medication class"""

    def __init__(self, patient_class: Patient, med_id: str):
        """Instantiate object

        Args:
            patient_class (Patient): Patient taking that medication
            med_id (str): Unique id of medication
        """
        super().__init__("med", med_id)
        self.patient = patient_class
        self.data = self.patient.medications_df[
            self.patient.medications_df["med_id"] == self.id
        ]
        # start and end available
        if len(self.data) == 2:
            self.start_date = self.data[self.data["is_start"] == 1]["date"].item()
            self.end_date = self.data[self.data["is_start"] == 0]["date"].item()
        # only start available
        else:
            self.start_date = self.data["date"].item()
            self.end_date = pd.NaT
        self.interval = [self.start_date, self.end_date]
        # self.get_related_visits()
        # TODO implement flags for missing data
        self.flag = None
        return

    def get_related_visits(self, tol: int = 0):
        """Retrieve visits close to medication

        Args:
            tol (int, optional): Tolerance to find close visits (in days). Defaults to 0.
        """
        self.visit_start = []
        self.visit_end = []
        self.visit_during = []
        for visit in self.patient.visits:
            # self.is_in_interval(visit)
            self.is_close_to_interval(visit, tol)
        return

    def is_close_to_interval(self, visit_class: Visit, tol: int = 0):
        """Determine if visit is close to [start, stop] interval of medication

        Args:
            visit_class (Visit): visit to consider
            tol (int, optional): Tolerance (in days) to consider visit close to medication. Defaults to 0.
        """
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
