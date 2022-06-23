from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.data_objects.patient import Patient

from tqdm import tqdm


class DatasetMultitask(Dataset):
    def __init__(
        self,
        device: str,
        df_dict: dict,
        ids: list,
        target_names: list,
        event_names: list,
        min_num_targets: int,
        mapping=None,
    ):
        """Instantiate object.

        Args:
            device (str): CPU or GPU
            df_dict (dict): Dictionnary of datframes
            ids (list): Patient ids to keep
            target_category_name (str): Name of categorical target
            event_names (list): Names of possible events (e.g. visit, medication)
            min_num_targets (int): Minimum number of visits to keep patient
            mapping (_type_, optional): Mapping used for categorical features. Defaults to None.
        """
        self.initial_df_dict = df_dict
        self.patient_ids = list(ids)
        self.target_names = target_names
        self.event_names = event_names
        self.min_num_targets = min_num_targets
        self.device = device
        self.instantiate_patients()
        self.target_category_name = "None"
        if mapping is not None:
            self.mapping = mapping

        return
