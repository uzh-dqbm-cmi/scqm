# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import random
import itertools
import sys

sys.path.append("../scqm")
import numpy as np
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.preprocessing.select_features import extract_adanet_features
import torch
import pandas as pd
import copy
from scqm.test_bed.fake_scqm import get_df_dict
from scqm.custom_library.cv.adaptive_net_multi import CVAdaptivenet
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.models.other_net_with_attention import (
    OthernetOptimizedWithAttention,
)
from scqm.custom_library.models.other_net import OthernetOptimized
from scqm.custom_library.models.adaptive_net import Adaptivenet

import torch.multiprocessing as mp

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
        patients_df,
        medications_df,
        visits_df,
        targets_df,
        _,
        _,
        haq_df,
        _,
    ) = extract_adanet_features(
        df_dict_processed,
        transform_meds=True,
        das28=True,
        only_meds=True,
        joint_df=False,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": patients_df,
        "med": medications_df,
        "targets": targets_df,
        "haq": haq_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_visits = 2
    # instantiate dataset
    dataset = Dataset(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        "das28_increase",
        ["a_visit", "med", "haq"],
        min_num_visits,
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
    cv = CVAdaptivenet(dataset, k=5)

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
    fold = 1
    print(f"fold {fold}")

    combinations = list(itertools.product(*parameters.values()))
    self = cv
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
    combinations = random.sample(combinations, 3)
    processes = []
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
            n_epochs=2,
            batch_size=int(len(self.dataset) / 15),
            lr=lr,
            balance_classes=bal,
            use_early_stopping=False,
        )
        # TODO re-check
        # trainer.train_model(model, self.partition, debug_patient=False)
        p = mp.Process(target=trainer.train_model, args=(model, self.partition, False))
        p.start()
        processes.append(p)
    print("End of script")
