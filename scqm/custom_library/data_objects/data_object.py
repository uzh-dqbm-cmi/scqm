class DataObject:
    def __init__(self, df_dict, patient_id):
        for name in df_dict.keys():
            setattr(
                self,
                str(name) + "_df",
                df_dict[name][df_dict[name]["patient_id"] == patient_id],
            )
        return

    def add_info(self, name, value):
        setattr(self, name, value)
        return
