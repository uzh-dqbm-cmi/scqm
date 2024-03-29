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

    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    with open("/cluster/work/medinfmk/scqm/tmp/saved_cv_cpu_17_08.pickle", "rb") as f:
        cv = pickle.load(f)
    # df_dict = load_dfs_all_data()
    # df_dict_pro = preprocessing(df_dict)
    # (
    #     general_df,
    #     med_df,
    #     visits_df,
    #     basdai_df,
    #     targets_df_das28,
    #     targets_df_basdai,
    #     socioeco_df,
    #     radai_df,
    #     haq_df,
    #     mny_df,
    # ) = extract_multitask_features(df_dict_pro, transform_meds=True, only_meds=True)
    # df_dict_ada = {
    #     "a_visit": visits_df,
    #     "patients": general_df,
    #     "med": med_df,
    #     "targets_das28": targets_df_das28,
    #     "targets_basdai": targets_df_basdai,
    #     "socio": socioeco_df,
    #     "radai": radai_df,
    #     "haq": haq_df,
    #         "basdai": basdai_df,
    #         "mny": mny_df,
    # }
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # min_num_targets = 2
    # # instantiate dataset
    # events = ["a_visit", "med", "socio", "radai", "haq", "basdai", "mny"]
    # dataset = DatasetMultitask(
    #     device,
    #     df_dict_ada,
    #     df_dict_ada["patients"]["patient_id"].unique(),
    #     ["das283bsr_score", "basdai_score"],
    #     events,
    #     min_num_targets,
    #     )
    # # random.sample(list(df_dict_ada["patients"]["patient_id"].unique()), 4000)
    # dataset.drop(
    #      [
    #           id_
    #           for id_, patient in dataset.patients.items()
    #           if len(patient.visit_ids) <= 2
    #           ]
    #      )
    # print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    # dataset.get_masks()
    # with open("/opt/tmp/dataset_cpu_17_08.pickle", "wb") as handle:
    #     pickle.dump(dataset, handle)
    # dataset.create_dfs()
    # dataset.transform_to_numeric_adanet()

    # cv = CVMultitask(dataset, k=5)
    # with open("/opt/tmp/saved_cv_cpu_17_08.pickle", "wb") as f:
    #     pickle.dump(cv, f)
    dataset = cv.dataset
    partition = cv.partition
    # dataset.move_to_device("cpu")
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

    processes = []
    Folds = [0, 1]
    for fold in Folds:

        print(fold)
        device = torch.device("cuda:" + str(fold) if torch.cuda.is_available() else "cpu")
        cv.partition.set_current_fold(fold)
        model_specifics["device"] = device
        model = Multitask(model_specifics, device)
        trainer = MultitaskTrainer(
            model,
            dataset,
            n_epochs=2,
            batch_size={
                "das28": int(len(dataset) / 15),
                "basdai": int(len(dataset) / (15 * 3)),
            },
            lr=1e-2,
            balance_classes=True,
            use_early_stopping=False,
        )

        p = mp.Process(target=trainer.train_model, args=(model, cv.partition, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("End of script")
