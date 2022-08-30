from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
import copy
import pandas as pd
import torch
import random
import pickle
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition

from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing
import torch.multiprocessing as mp


if __name__ == "__main__":
    df_dict = load_dfs_all_data()
    df_dict_pro = preprocessing(df_dict)
    (
        general_df,
        med_df,
        visits_df,
        basdai_df,
        targets_df_das28,
        targets_df_basdai,
        socioeco_df,
        radai_df,
        haq_df
    ) = extract_multitask_features(df_dict_pro, transform_meds=True, only_meds=True)
    df_dict_ada = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "targets_das28": targets_df_das28,
        "targets_basdai": targets_df_basdai,
        "socio": socioeco_df,
        "radai": radai_df,
        "haq": haq_df,
        "basdai": basdai_df
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    events = ["a_visit", "med", "socio", "radai", "haq", "basdai"]
    dataset = DatasetMultitask(
        device,
        df_dict_ada,
        df_dict_ada["patients"]["patient_id"].unique(),
        ["das283bsr_score", "basdai_score"],
        events,
        min_num_targets,
        )
    # random.sample(list(df_dict_ada["patients"]["patient_id"].unique()), 4000)
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
            ]
        )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks()
    with open("/opt/tmp/dataset_cpu_25_08.pickle", "wb") as handle:
        pickle.dump(dataset, handle)
    dataset.create_dfs()
    dataset.transform_to_numeric_adanet()

    cv = CVMultitask(dataset, k=5)
    with open("/opt/tmp/saved_cv_cpu_25_08.pickle", "wb") as f:
        pickle.dump(cv, f)
