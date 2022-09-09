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
import shap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed = 0
    # create fake data
    df_dict = get_df_dict(num_patients=200)
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
    target_name = "asdas_score"
    if target_name == "das283bsr_score":
        input_size = dataset.joint_das28_df_scaled_tensor_train.shape[1]
    elif target_name == "asdas_score":
        input_size = dataset.joint_asdas_df_scaled_tensor_train.shape[1]
    config = {"input_size": input_size, "output_size": 1}
    model = MLP(config, device)
    trainer = MLPTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size=30,
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
        target_name=target_name,
    )
    trainer.train_model(model, partition)

    # # try shap

    # x_train = dataset.joint_das28_df_scaled_tensor_train[:100]
    # x_test = dataset.joint_das28_df_scaled_tensor_test

    # def model_wrapper(x):
    #     x_tensor = torch.tensor(x.values)
    #     return trainer.model(x_tensor.float()).flatten()

    # with torch.no_grad():

    #     # explainer = shap.Explainer(model_wrapper, np.array(x_train))
    #     explainer = shap.Explainer(
    #         model_wrapper,
    #         pd.DataFrame(
    #             np.array(x_train), columns=dataset.joint_das28_df_columns_in_tensor
    #         ),
    #     )
    #     shap_values = explainer(
    #         pd.DataFrame(
    #             np.array(x_test), columns=dataset.joint_das28_df_columns_in_tensor
    #         )
    #     )

    # # "all" data points
    # plt.figure(figsize=(10, 10))
    # shap.summary_plot(
    #     shap_values,
    #     show=False,
    # )
    # plt.savefig("shap_all.png", bbox_inches="tight")
    # # single data point
    # plt.figure(figsize=(10, 10))
    # shap.plots.waterfall(shap_values[2], show=False)
    # plt.savefig("shap_waterfall.png", bbox_inches="tight")

    # plt.figure(figsize=(10, 10))
    # shap.plots.force(shap_values[2], matplotlib=True, show=False)
    # plt.savefig("shap_force.png", bbox_inches="tight")
    # plt.figure(figsize=(10, 10))
    # shap.plots.bar(shap_values, show=False)
    # plt.savefig("shap_mean.png", bbox_inches="tight")

    # plt.figure(figsize=(10, 10))
    # shap.plots.scatter(shap_values[:, "das283bsr_score"], color=shap_values)
    # plt.savefig("shap_scatter.png", bbox_inches="tight")

    # plt.figure(figsize=(10, 10))
    # shap.plots.heatmap(shap_values)
    # plt.savefig("shap_heatmap.png", bbox_inches="tight")

    # print("end")
