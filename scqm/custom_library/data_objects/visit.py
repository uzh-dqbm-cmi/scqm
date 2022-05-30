from scqm.custom_library.data_objects.event import Event
from scqm.custom_library.data_objects.patient import Patient


class Visit(Event):
    """Visit class"""

    def __init__(
        self,
        patient_class: Patient,
        visit_id: str,
        date: str,
        target: str = "das283bsr_score",
    ):
        """Instantiate a given visit object

        Args:
            patient_class (Patient): Corresponding patient object
            visit_id (str): Unique visit identifier
            date (str): date of visit
            target (str, optional): Feature to be used as target. Defaults to "das283bsr_score".
        """
        super().__init__("a_visit", visit_id)
        self.patient = patient_class
        self.date = date
        self.data = self.patient.visits_df[
            self.patient.visits_df["uid_num"] == visit_id
        ]
        if target:
            self.target = self.data[target]
        self.med_start = []
        self.med_during = []
        self.med_end = []
        return
