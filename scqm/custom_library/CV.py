from models import Adaptivenet
from training import AdaptivenetTrainer
from partition import DataPartition
from preprocessing import (
    load_dfs,
    load_dfs_all_data,
    preprocessing,
    extract_adanet_features,
)
from data_objects import Dataset
import itertools
import numpy as np
import sys
import csv
import random
import time
import os
import torch
import pickle
import gc

from utils import set_seeds


# TODO early stopping as parameter
# TODO shell script for cv
# TODO cleaner preprocessing
# TODO testing module
# TODO modularise baseline
# TODO stratifier
# TODO decoders in AE style
# TODO optimize metrics computation (ie something to easily just log losses)


class CVWrapper:
    def __init__(self, dataset, k):
        set_seeds(0)
        self.dataset = dataset
        self.k = k
        self.partition = DataPartition(self.dataset, k=self.k)

    def set_grid(self, parameters: dict):
        self.parameter_names = list(parameters.keys())
        self.parameters = parameters

    def perform_cv(self):
        raise NotImplementedError


class CVAdaptivenet(CVWrapper):
    def perform_cv(self, fold, n_epochs=400, search="random", num_combi=1):
        combinations = list(itertools.product(*self.parameters.values()))
        self.partition.set_current_fold(fold)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        task = "regression"

        if task == "regression":
            num_targets = 1
        else:
            num_targets = 3

        # instantiate model
        num_feature_dict = {
            event: getattr(self.dataset, event + "_df_scaled_tensor_train").shape[1]
            for event in self.dataset.event_names
        }
        size_out_dict = {
            event: int(num_feature_dict[event] / 10) + 1
            for event in self.dataset.event_names
        }
        num_feature_dict["patients"] = getattr(
            self.dataset, "patients" + "_df_scaled_tensor_train"
        ).shape[1]
        size_out_dict["patients"] = int(num_feature_dict["patients"] / 10) + 1
        batch_first = True
        # random search instead of grid search
        if search == "random":
            combinations = random.sample(combinations, num_combi)
        for ind, (
            size_embedding,
            num_layers_enc,
            hidden_enc,
            size_history,
            num_layers,
            num_layers_pred,
            hidden_pred,
            lr,
            p,
            bal,
        ) in enumerate(combinations):
            # to save
            path = "/cluster/home/ctrottet/runs/scqm/" + time.strftime("%Y%m%d-%H%M")
            os.mkdir(path)

            print(f"{ind} combination out of {len(combinations)}")
            print(
                f"size_embedding {size_embedding}, num_layers_enc {num_layers_enc}, hidden_enc {hidden_enc}, size_history {size_history}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, dropout {p}, lr {lr}"
            )
            model_specifics = {
                "task": task,
                "num_targets": num_targets,
                "size_embedding": size_embedding,
                "num_layers_enc": num_layers_enc,
                "hidden_enc": hidden_enc,
                "size_history": size_history,
                "num_layers": num_layers,
                "num_layers_pred": num_layers_pred,
                "hidden_pred": hidden_pred,
                "event_names": self.dataset.event_names,
                "num_general_features": self.dataset.patients_df_scaled_tensor_train.shape[
                    1
                ],
                "dropout": p,
                "batch_first": batch_first,
                "device": device,
                "model_type": "padd",
            }
            for key in num_feature_dict:
                model_specifics[key] = {
                    "num_features": num_feature_dict[key],
                    "size_out": size_out_dict[key],
                }
            model_specifics["size_embedding"] = max(
                [model_specifics[key]["size_out"] for key in num_feature_dict]
            )
            model = Adaptivenet(model_specifics, device)
            self.dataset.min_num_visits = 2
            trainer = AdaptivenetTrainer(
                model,
                self.dataset,
                n_epochs,
                batch_size=int(len(self.dataset) / 15),
                lr=lr,
                balance_classes=bal,
                use_early_stopping=False,
            )
            gc.collect()
            trainer.train_model(model, self.partition, debug_patient=False)
            # for memory
            delattr(trainer, "dataset")
            with open(path + "/params.pkl", "wb") as f:
                pickle.dump(model_specifics, f)
            with open(path + "/trainer.pkl", "wb") as f:
                pickle.dump(trainer, f)


if __name__ == "__main__":

    model = str(sys.argv[1])
    reload = str(sys.argv[3])
    if reload == "True":
        if model == "adanet":
            with open("/opt/data/processed/saved_cv_ada.pickle", "rb") as handle:
                cv = pickle.load(handle)
        else:
            with open("/opt/data/processed/saved_cv.pickle", "rb") as handle:
                cv = pickle.load(handle)
    else:
        if model == "adanet":
            with open("/opt/data/processed/saved_dataset.pickle", "rb") as handle:
                dataset = pickle.load(handle)
        else:
            with open(
                "/opt/data/processed/saved_dataset_more_features.pickle", "rb"
            ) as handle:
                dataset = pickle.load(handle)
        # prepare for training
        dataset.create_dfs()
        if model == "adanet":
            dataset.transform_to_numeric_adanet()
            cv = CVAdaptivenet(dataset, k=5)
            with open("/opt/data/processed/saved_cv_ada.pickle", "wb") as f:
                pickle.dump(cv, f)
        else:
            dataset.transform_to_numeric()
            cv = CVAdaptivenet(dataset, k=5)
            with open("/opt/data/processed/saved_cv.pickle", "wb") as f:
                pickle.dump(cv, f)

    parameters = {
        "SIZE_EMBEDDING": np.array([3, 5]),
        "NUM_LAYERS_ENC": np.array([2, 4]),
        "HIDDEN_ENC": np.array([100]),
        "SIZE_HISTORY": np.array([10, 20]),
        "NUM_LAYERS": np.array([2]),
        "NUM_LAYERS_PRED": np.array([2]),
        "HIDDEN_PRED": np.array([100]),
        "LR": np.array([1e-2]),
        "P": np.array([0.1, 0.2]),
        "BALANCE_CLASSES": np.array([True]),
    }
    cv.set_grid(parameters)
    fold = int(sys.argv[2])
    print(f"fold {fold}")
    cv.perform_cv(fold=fold, search="random", n_epochs=50, num_combi=6)
