import sys

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
import numpy as np
import sys
import itertools
import os
from scqm.custom_library.parameters.cv import get_parameters
import datetime

if __name__ == "__main__":

    fold = int(sys.argv[1])
    print(fold)
    date = datetime.datetime.now().strftime("%d_%m_%Y")
    print(f"creating directory")
    path = "/cluster/work/medinfmk/scqm/tmp/fold" + str(fold) + "/" + date
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    print("loading data")

    with open("/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint.pickle", "rb") as f:
        cv = pickle.load(f)

    dataset = cv.dataset
    partition = cv.partition
    print(f"Starting job on fold {fold}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cv.partition.set_current_fold(fold)
    # parameters that are not changed throughout the folds
    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    with open("/cluster/work/medinfmk/scqm/tmp/params.pickle", "rb") as f:
        cvparams, combinations = pickle.load(f)
    # cvparams, combinations = get_parameters(fold, int(sys.argv[2]))
    name_mapping = {key: index for index, key in enumerate(list(cvparams.keys()))}
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
            n_epochs=100,
            batch_size={
                "das28": int(len(dataset) / 15),
                "asdas": int(len(dataset) / (15 * 4.5)),
            },
            lr=params[name_mapping["lr"]],
            balance_classes=True,
            use_early_stopping=False,
        )
        trainer.train_model(model, cv.partition, debug_patient=False)
        trainer.best_loss_valid
        result_dict[params] = (trainer.best_loss_valid, trainer.optimal_epoch)

        delattr(trainer, "dataset")
        with open(
            path + "/trainer_" + str(fold) + "_" + str(index) + ".pickle",
            "wb",
        ) as handle:
            pickle.dump(trainer, handle)

    with open(
        path + "/results_" + str(fold) + ".pickle",
        "wb",
    ) as handle:
        pickle.dump(result_dict, handle)

    print("End of script")
