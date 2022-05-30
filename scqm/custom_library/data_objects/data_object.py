class DataObject:
    """
    Base class to handle data on the patient level.
    """

    def __init__(self, df_dict: dict, patient_id: str):
        """Instantiate object with attributes corresponding to the data related to a given patient.

        Args:
            df_dict (dict): Dictionnary of dataframes
            patient_id (str): Unique identifier of a patient
        """
        for name in df_dict.keys():
            setattr(
                self,
                str(name) + "_df",
                df_dict[name][df_dict[name]["patient_id"] == patient_id],
            )
        return

    def add_info(self, name: str, value) -> None:
        """Set new attribute

        Args:
            name (str): name of attribute
            value (_type_): value of attribute
        """
        setattr(self, name, value)
        return
