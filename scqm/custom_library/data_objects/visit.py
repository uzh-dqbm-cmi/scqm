from scqm.custom_library.data_objects.event import Event

class Visit(Event):
    def __init__(self, patient_class, visit_id, date, target="das283bsr_score"):
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
