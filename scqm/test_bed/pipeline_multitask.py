# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import sys
sys.path.append("../scqm")

from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
import time
import torch
import pandas as pd
import copy
from scqm.test_bed.fake_scqm import get_df_dict
from scqm.custom_library.trainers.multiloss import MultilossTrainer
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.models.other_net_with_double_attention import (
    OthernetWithDoubleAttention,
)
from scqm.custom_library.models.other_net_multiloss import OthernetMultiloss
from scqm.custom_library.models.other_net_with_attention import (
    OthernetWithAttention,
)
from scqm.custom_library.models.other_net import Othernet
from scqm.custom_library.models.adaptive_net import Adaptivenet

import cProfile
import pstats




# setting path




# setting path

if __name__ == "__main__":
    # create fake data
    df_dict = get_df_dict(num_patients=100)
    real_data = False
    df_dict_processed = copy.deepcopy(df_dict)
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(
            pd.to_datetime
        )
    (
        general_df,
        med_df,
        visits_df,
        basdai_df,
        targets_df,
        socioeco_df,
        radai_df,
        haq_df,
    ) = extract_multitask_features(
        df_dict_processed,
        transform_meds=True,
        das28_or_basdai=True,
        only_meds=True,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "targets": targets_df,
        "haq": haq_df,
        "basdai": basdai_df
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_visits = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        "das28_increase",
        ["a_visit", "med", "haq", "basdai"],
        min_num_visits,
    )
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print('End of script')