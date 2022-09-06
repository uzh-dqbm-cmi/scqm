import sys

sys.path.append("../scqm")
import numpy as np
from scqm.custom_library.clustering.utils import (
    get_features,
    get_histories_and_features,
)
from scqm.custom_library.clustering.similarity import compute_similarity
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
import time
import torch
import pandas as pd
import copy
from scqm.test_bed.fake_scqm import get_df_dict
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
from scqm.custom_library.trainers.mlp import MLPTrainer
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.models.MLP import MLP
from statistics import mean
from sklearn.cluster import KMeans
import pstats
import cProfile
from tqdm import tqdm


if __name__ == "__main__":
    seed = 0
    # create fake data
    df_dict = get_df_dict(num_patients=500)
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
        targets_df_das28,
        targets_df_asdas,
        socioeco_df,
        radai_df,
        haq_df,
        joint_df,
    ) = extract_multitask_features(
        df_dict_processed,
        transform_meds=True,
        only_meds=True,
        real_data=False,
        joint_df=True,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "targets_das28": targets_df_das28,
        "targets_asdas": targets_df_asdas,
        "haq": haq_df,
        "joint": joint_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        ["das283bsr_score", "asdas_score"],
        ["a_visit", "med", "haq"],
        min_num_targets,
    )
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks()
    dataset.post_process_joint_df()
    dataset.create_dfs()
    # get targets for baseline
    # prepare for training
    dataset.transform_to_numeric_adanet(real_data)
    partition = MultitaskPartition(dataset, k=3)
    fold = int(0)
    partition.set_current_fold(fold)

    input_size = dataset.joint_das28_df_scaled_tensor_train.shape[1]
    config = {"input_size": input_size, "output_size": 1}
    model = MLP(config, device)
    trainer = MLPTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size=10,
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
        target_name="das283bsr_score",
    )
    trainer.train_model(model, partition)
    # train

    # profiler = cProfile.Profile()
    # profiler.enable()
    # start = time.time()
    # trainer.train_model(model, partition, debug_patient=False)
    # end = time.time()
    # print(end - start)
    # train baseline
