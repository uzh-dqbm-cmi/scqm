import sys

sys.path.append("../scqm")

import pickle
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
import time
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
import pandas as pd
import torch
import numpy as np
import random
import itertools
from scqm.custom_library.models.adaptive_net import Adaptivenet
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.test_bed.fake_scqm import get_df_dict
import copy
from scqm.custom_library.parameters.cv import get_parameters


# setting path

if __name__ == "__main__":
    device = "cpu"
    seed = 0
    fold = int(sys.argv[1])
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
        basdai_df,
        targets_df_das28,
        targets_df_basdai,
        socioeco_df,
        radai_df,
        haq_df,
        mny_df,
    ) = extract_multitask_features(
        df_dict_processed,
        transform_meds=True,
        only_meds=True,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "targets_das28": targets_df_das28,
        "targets_basdai": targets_df_basdai,
        "haq": haq_df,
        "basdai": basdai_df,
    }
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        "cpu",
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        ["das283bsr_score", "basdai_score"],
        ["a_visit", "med", "haq", "basdai"],
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
    dataset.create_dfs()
    # prepare for training
    dataset.transform_to_numeric_adanet(real_data)
    cv = CVMultitask(dataset, k=5)
    cv.partition.set_current_fold(fold)
    # with open("scqm/test_bed/dummy_data/params", "rb") as f:
    #     cvparams, combinations = pickle.load(f)
    cvparams, combinations = get_parameters(fold, 3)
    print(combinations)
    name_mapping = {key: index for index, key in enumerate(list(cvparams.keys()))}
    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    result_dict = {comb: np.nan for comb in combinations}
    for index, params in enumerate(combinations):
        print(f"{index + 1} combination out of {len(combinations)}")

        size_out_dict = {
            event: int(num_feature_dict[event] / params[name_mapping["size_out_scale"]])
            + 1
            for event in dataset.event_names
        }

        size_out_dict["patients"] = (
            int(num_feature_dict["patients"] / params[name_mapping["size_out_scale"]])
            + 1
        )
        # size_history_dict = {event: num_feature_dict[event]*2 for event in dataset.event_names}
        model_specifics = {
            "num_layers_enc": params[name_mapping["num_layers_enc"]],
            "hidden_enc": params[name_mapping["hidden_enc"]],
            "size_history": params[name_mapping["size_history"]],
            "num_layers": params[name_mapping["num_layers"]],
            "num_layers_pred": params[name_mapping["num_layers_pred"]],
            "hidden_pred": params[name_mapping["hidden_pred"]],
            "event_names": dataset.event_names,
            "num_general_features": dataset.patients_df_scaled_tensor_train.shape[1],
            "dropout": params[name_mapping["dropout"]],
            "batch_first": True,
            "device": device,
        }
        for key in num_feature_dict:
            model_specifics[key] = {
                "num_features": num_feature_dict[key],
                "size_out": size_out_dict[key],
            }
        model_specifics["size_embedding"] = max(
            [model_specifics[key]["size_out"] for key in num_feature_dict]
        )

        model = Multitask(model_specifics, device)
        trainer = MultitaskTrainer(
            model,
            dataset,
            n_epochs=10,
            batch_size={
                "das28": int(len(dataset) / 15),
                "basdai": int(len(dataset) / (15 * 3)),
            },
            lr=1e-2,
            balance_classes=True,
            use_early_stopping=False,
        )
        trainer.train_model(model, cv.partition, debug_patient=False)
        result_dict[params] = (trainer.best_loss_valid, trainer.optimal_epoch)

        delattr(trainer, "dataset")
        with open(
            "scqm/test_bed/dummy_data/trainer_" + str(fold) + "_" + str(index), "wb"
        ) as handle:
            pickle.dump(trainer, handle)
