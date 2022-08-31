import os
import sys
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
import copy
import pandas as pd
import torch
import time
import pickle
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition

from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing

if __name__ == "__main__":
    reload = True
    if reload:
        with open("/opt/tmp/saved_cv_asdas_without_basdai.pickle", "rb") as f:
            cv = pickle.load(f)
    else:
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
            haq_df,
            mny_df,
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
            "basdai": basdai_df,
            "mny": mny_df,
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        min_num_targets = 2
        # instantiate dataset
        events = ["a_visit", "med", "socio", "radai", "haq", "basdai", "mny"]
        dataset = DatasetMultitask(
            device,
            df_dict_ada,
            df_dict_ada["patients"]["patient_id"].unique(),
            ["das283bsr_score", "basdai_score"],
            events,
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
        with open("/opt/tmp/dataset_multitask_mny_09_08.pickle", "wb") as handle:
            pickle.dump(dataset, handle)
        dataset.create_dfs()
        dataset.transform_to_numeric_adanet()

        cv = CVMultitask(dataset, k=5)
        with open("/opt/tmp/saved_cv_multitask_mny_09_08.pickle", "wb") as f:
            pickle.dump(cv, f)
    dataset = cv.dataset
    partition = cv.partition
    print("start")
    fold = int(0)
    partition.set_current_fold(fold)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"fold {fold}")

    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    size_out_dict = {
        event: int(num_feature_dict[event] / 10) + 1 for event in dataset.event_names
    }
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    size_out_dict["patients"] = int(num_feature_dict["patients"] / 10) + 1
    # size_history_dict = {event: num_feature_dict[event]*2 for event in dataset.event_names}
    model_specifics = {
        "num_layers_enc": 2,
        "hidden_enc": 100,
        "size_history": 10,
        "num_layers": 1,
        "num_layers_pred": 2,
        "hidden_pred": 100,
        "event_names": dataset.event_names,
        "num_general_features": dataset.patients_df_scaled_tensor_train.shape[1],
        "dropout": 0.0,
        "batch_first": True,
        "device": device,
    }
    for key in num_feature_dict:
        model_specifics[key] = {
            "num_features": num_feature_dict[key],
            "size_out": size_out_dict[key],
        }
    model_specifics["size_history"] = 30
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )

    model = Multitask(model_specifics, device)
    # n_epochs = 60
    trainer = MultitaskTrainer(
        model,
        dataset,
        n_epochs=100,
        batch_size={
            "das28": int(len(dataset) / 15),
            "asdas": int(len(dataset) / (15 * 4.5)),
        },
        lr=1e-3,
        balance_classes=True,
        use_early_stopping=False,
    )
    # lr 1e-3, epochs 100
    trainer.train_model(model, partition, debug_patient=False)
    # store histories
    # subset = dataset.train_ids
    # numbers_of_target = [
    #     torch.sum(
    #         dataset.masks.available_target_mask[dataset.mapping_for_masks[patient]]
    #         == True
    #     ).item()
    #     for patient in subset
    # ]
    # histories = torch.empty(size=(sum(numbers_of_target), model.pred_input_size))
    # index_in_history = 0
    # for index, patient in enumerate(subset):
    #     _, _, _, hist = model.apply(dataset, patient, return_history=True)
    #     histories[index_in_history : index_in_history + numbers_of_target[index]] = hist
    #     index_in_history += numbers_of_target[index]
    delattr(trainer, "dataset")
    with open("/opt/tmp/trainer_asdas_without_basdai.pickle", "wb") as handle:
        pickle.dump(trainer, handle)
    # best "/opt/tmp/trainer_multitarget_09_08.pickle"
    # with open("/opt/tmp/train_histories.pickle", "wb") as handle:
    #     pickle.dump(histories, handle)
