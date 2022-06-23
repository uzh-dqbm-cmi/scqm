from __future__ import annotations
from scqm.custom_library.data_objects.event import Event

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scqm.custom_library.data_objects.patient import Patient


class Basdai(Event):
    """Visit class"""

    def __init__(
        self,
        patient_class: Patient,
        event_id: str,
        date: str,
        target: str = "basdai_score",
    ):
        """Instantiate a given visit object

        Args:
            patient_class (Patient): Corresponding patient object
            visit_id (str): Unique visit identifier
            date (str): date of visit
            target (str, optional): Feature to be used as target. Defaults to "das283bsr_score".
        """
        super().__init__("basdai", event_id)
        self.patient = patient_class
        self.date = date
        self.data = self.patient.basdai_df[
            self.patient.basdai_df["event_id"] == event_id
        ]
        if target:
            self.target = self.data[target]
        return
