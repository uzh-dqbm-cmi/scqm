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
from scqm.custom_library.utils import set_seeds


if __name__ == "__main__":
    set_seeds(0)
    with open("/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint.pickle", "rb") as f:
        old_cv = pickle.load(f)
    old_data = old_cv.dataset
    indices_dict = {
        "train_ids": old_data.train_ids,
        "valid_ids": old_data.valid_ids,
        "test_ids": old_data.test_ids,
    }
    # df_dict = load_dfs_all_data()
    # df_dict_pro = preprocessing(df_dict)
    # (
    #     general_df,
    #     med_df,
    #     visits_df,
    #     targets_df_das28,
    #     targets_df_asdas,
    #     socioeco_df,
    #     radai_df,
    #     haq_df,
    #     joint_df,
    # ) = extract_multitask_features(
    #     df_dict_pro, transform_meds=True, only_meds=True, joint_df=True
    # )
    # df_dict_ada = {
    #     "a_visit": visits_df,
    #     "patients": general_df,
    #     "med": med_df,
    #     "targets_das28": targets_df_das28,
    #     "targets_asdas": targets_df_asdas,
    #     "socio": socioeco_df,
    #     "radai": radai_df,
    #     "haq": haq_df,
    #     "joint": joint_df,
    # }
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # min_num_targets = 2
    # # instantiate dataset
    # events = ["a_visit", "med", "socio", "radai", "haq"]
    # dataset = DatasetMultitask(
    #     device,
    #     df_dict_ada,
    #     df_dict_ada["patients"]["patient_id"].unique(),
    #     ["das283bsr_score", "asdas_score"],
    #     events,
    #     min_num_targets,
    # )

    # # random.sample(list(df_dict_ada["patients"]["patient_id"].unique()), 4000)
    # dataset.drop(
    #     [
    #         id_
    #         for id_, patient in dataset.patients.items()
    #         if len(patient.visit_ids) <= 2
    #     ]
    # )
    # print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    # dataset.get_masks()
    # with open(
    #     "/cluster/work/medinfmk/scqm/tmp/dataset_with_joint_15_09.pickle", "wb"
    # ) as handle:
    #     pickle.dump(dataset, handle)
    with open(
        "/cluster/work/medinfmk/scqm/tmp/dataset_with_joint_15_09.pickle", "rb"
    ) as handle:
        dataset = pickle.load(handle)
    dataset.post_process_joint_df()
    dataset.create_dfs()
    dataset.transform_to_numeric_adanet()

    cv = CVMultitask(dataset, k=5, split_indices=indices_dict)
    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_15_09.pickle", "wb"
    ) as f:
        pickle.dump(cv, f)
