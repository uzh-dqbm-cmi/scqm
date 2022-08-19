# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
from statistics import mean
from sklearn.cluster import KMeans
import pstats
import cProfile
from tqdm import tqdm

import sys

sys.path.append("../scqm")
from scqm.custom_library.models.multitask_net import Multitask

from scqm.custom_library.trainers.multitask_net import MultitaskTrainer

from scqm.test_bed.fake_scqm import get_df_dict
import copy
import pandas as pd
import torch
import time
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition

from scqm.custom_library.clustering.similarity import compute_similarity
from scqm.custom_library.clustering.utils import (
    get_features,
    get_histories_and_features,
)

import numpy as np

# setting path

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        device,
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
    partition = MultitaskPartition(dataset, k=3)
    fold = int(0)
    partition.set_current_fold(fold)
    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    size_out_dict = {event: 2 for event in dataset.event_names}
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    size_out_dict["patients"] = 2
    model_specifics = {
        "task": "regression",
        "num_targets": 1,
        "size_embedding": 2,
        "num_layers_enc": 2,
        "hidden_enc": 10,
        "size_history": 2,
        "num_layers": 1,
        "num_layers_pred": 2,
        "hidden_pred": 10,
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
    model_specifics["size_history"] = 3
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )
    model = Multitask(model_specifics, device)
    trainer = MultitaskTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size={"das28": int(len(dataset) / 10), "basdai": int(len(dataset) / 10)},
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )

    # train

    # profiler = cProfile.Profile()
    # profiler.enable()
    start = time.time()
    trainer.train_model(model, partition, debug_patient=False)
    end = time.time()
    print(end - start)
    # test apply function
    for p in dataset.test_ids:
        if dataset[p].target_name in ["both", "das283bsr_score"]:
            model.apply(dataset, p, "das283bsr_score", return_history=False)
        if dataset[p].target_name in ["both", "basdai_score"]:
            model.apply(dataset, p, "basdai_score", return_history=False)
    # kmeans clustering
    subset = dataset.train_ids
    subset_das28 = [
        p for p in subset if dataset[p].target_name in ["both", "das283bsr_score"]
    ]
    subset_basdai = [p for p in subset if dataset[p].target_name == "basdai_score"]

    # torch.sum(dataset.masks.available_target_mask == True).item()
    numbers_of_target = [
        torch.sum(
            dataset.masks_das28.available_target_mask[
                dataset.mapping_for_masks_das28[patient]
            ]
            == True
        ).item()
        for patient in subset_das28
    ]
    numbers_of_target.extend(
        [
            torch.sum(
                dataset.masks_basdai.available_target_mask[
                    dataset.mapping_for_masks_basdai[patient]
                ]
                == True
            ).item()
            for patient in subset_basdai
        ]
    )
    histories = torch.empty(size=(sum(numbers_of_target), model.pred_input_size))
    raw_histories = torch.empty(
        size=(
            sum(numbers_of_target),
            sum([model.config[event]["num_features"] for event in dataset.event_names])
            + model.config["num_general_features"],
        )
    )
    raw_histories_unscaled = np.empty(
        shape=(
            sum(numbers_of_target),
            sum(
                [
                    getattr(dataset, event + "_df").shape[1]
                    for event in dataset.event_names
                ]
            )
            + dataset.patients_df.shape[1],
        ),
        dtype="object",
    )
    index_in_history = 0
    raw_features = {}
    raw_features_unscaled = {}
    for index, patient in enumerate(subset_das28):
        _, _, _, hist, hist_per_event, _, _ = model.apply(
            dataset, patient, "das283bsr_score", return_history=True
        )
        (
            raw_features[patient],
            raw_histories[
                index_in_history : index_in_history + numbers_of_target[index]
            ],
            raw_features_unscaled[patient],
            raw_histories_unscaled[
                index_in_history : index_in_history + numbers_of_target[index]
            ],
            _,
        ) = get_features(model, dataset, patient, "das283bsr_score")

        histories[index_in_history : index_in_history + numbers_of_target[index]] = hist
        index_in_history += numbers_of_target[index]
    for index, patient in enumerate(subset_basdai):
        _, _, _, hist, hist_per_event, _, _ = model.apply(
            dataset, patient, "basdai_score", return_history=True
        )
        (
            raw_features[patient],
            raw_histories[
                index_in_history : index_in_history
                + numbers_of_target[index + len(subset_das28)]
            ],
            raw_features_unscaled[patient],
            raw_histories_unscaled[
                index_in_history : index_in_history
                + numbers_of_target[index + len(subset_das28)]
            ],
            _,
        ) = get_features(model, dataset, patient, "basdai_score")
        histories[
            index_in_history : index_in_history
            + numbers_of_target[index + len(subset_das28)]
        ] = hist
        index_in_history += numbers_of_target[index + len(subset_das28)]
    k = 3
    print("KMeans")
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(histories)
    # evaluate on test
    subset_test = dataset.test_ids
    subset_test_das28 = [
        p for p in subset_test if dataset[p].target_name in ["both", "das283bsr_score"]
    ]
    subset_test_basdai = [
        p for p in subset_test if dataset[p].target_name == "basdai_score"
    ]
    numbers_of_target_test = [
        torch.sum(
            dataset.masks_das28.available_target_mask[
                dataset.mapping_for_masks_das28[patient]
            ]
            == True
        ).item()
        for patient in subset_test_das28
    ]
    numbers_of_target_test.extend(
        [
            torch.sum(
                dataset.masks_basdai.available_target_mask[
                    dataset.mapping_for_masks_basdai[patient]
                ]
                == True
            ).item()
            for patient in subset_test_basdai
        ]
    )
    histories_test = torch.empty(
        size=(sum(numbers_of_target_test), model.pred_input_size)
    )
    raw_histories_test = torch.empty(
        size=(
            sum(numbers_of_target_test),
            sum([model.config[event]["num_features"] for event in dataset.event_names])
            + model.config["num_general_features"],
        )
    )

    index_in_history = 0
    patient_in_embedding_test = {
        patient: {"indices": []} for patient in subset_test_das28 + subset_test_basdai
    }

    raw_features_test = {}
    for index, patient in enumerate(subset_test_das28):
        (
            predictions,
            target_values,
            time_to_targets,
            hist,
            hist_per_event,
            _,
            _,
        ) = model.apply(dataset, patient, "das283bsr_score", return_history=True)
        (
            raw_features_test[patient],
            raw_histories_test[
                index_in_history : index_in_history + numbers_of_target_test[index]
            ],
            _,
            _,
            _,
        ) = get_features(model, dataset, patient, "das283bsr_score")
        histories_test[
            index_in_history : index_in_history + numbers_of_target_test[index]
        ] = hist
        index_in_history += numbers_of_target_test[index]
        patient_in_embedding_test[patient]["indices"] = np.arange(
            index_in_history, index_in_history + numbers_of_target_test[index]
        )

    for index, patient in enumerate(subset_test_basdai):

        (
            predictions,
            target_values,
            time_to_targets,
            hist,
            hist_per_event,
            att,
            global_att,
        ) = model.apply(dataset, patient, "basdai_score", return_history=True)
        (
            raw_features_test[patient],
            raw_histories_test[
                index_in_history : index_in_history
                + numbers_of_target_test[index + len(subset_test_das28)]
            ],
            _,
            _,
            _,
        ) = get_features(model, dataset, patient, "basdai_score")
        histories_test[
            index_in_history : index_in_history
            + numbers_of_target_test[index + len(subset_test_das28)]
        ] = hist
        index_in_history += numbers_of_target_test[index + len(subset_test_das28)]
        patient_in_embedding_test[patient]["indices"] = np.arange(
            index_in_history,
            index_in_history + numbers_of_target_test[index + len(subset_test_das28)],
        )

    # kmeans
    kmeans.predict(histories_test)
    # MSE between patients
    similarities_mse = {}
    similarities_cos = {}
    for index, p in enumerate(subset_test_das28):
        if len(patient_in_embedding_test[p]["indices"]) > 0:
            similarities_mse[p] = compute_similarity(
                p,
                subset_test_das28,
                histories_test,
                patient_in_embedding_test,
                "mse",
            )
            print(similarities_mse[p][0][p])
            similarities_cos[p] = compute_similarity(
                p,
                subset_test_das28,
                histories_test,
                patient_in_embedding_test,
                "cosine",
            )
            print(similarities_cos[p][0][p])

    def aggregate_similarities(similarities, metric="mse"):
        # for each patient compute average mean similarity to each other patient {pj : {pi : average}}
        mean_similarities = {p: {} for p in similarities.keys()}
        visit_similarities = {
            p: {visit: {} for visit in range(len(similarities[p]))}
            for p in similarities.keys()
        }
        for pi in similarities.keys():
            for pj in similarities.keys():
                mean_simil = 0
                for visit in range(len(similarities[pi])):
                    simil = similarities[pi][visit][pj]
                    mean_simil += torch.mean(simil)
                    index_simil = (
                        torch.argmin(simil).item()
                        if metric == "mse"
                        else torch.argmax(simil).item()
                    )
                    visit_similarities[pi][visit][pj] = (
                        index_simil,
                        simil[index_simil],
                    )
                mean_similarities[pi][pj] = mean_simil / len(similarities[pi])

        # for each patient for each visit find most similar visit in each other patient history
        #  {pj:{v0:{pi: value, }}}

        return mean_similarities, visit_similarities

    def find_most_similar_patients(similarities, metric="mse"):

        most_similar_patients = {patient: [] for patient in tqdm(similarities)}
        for patient in similarities:
            values = similarities[patient].items()
            if metric == "mse":
                most_similar_patients[patient] = sorted(values, key=lambda x: x[1])
            else:
                most_similar_patients[patient] = sorted(
                    values, key=lambda x: x[1], reverse=True
                )

        return most_similar_patients

    def find_most_similar_visits(similarities, metric="mse"):
        most_similar_visits = {
            patient: {visit: [] for visit in range(len(similarities[patient]))}
            for patient in similarities
        }
        for patient in tqdm(similarities):
            for visit in similarities[patient]:
                values = similarities[patient][visit].items()
                if metric == "mse":
                    most_similar_visits[patient][visit] = sorted(
                        values, key=lambda x: x[1][1]
                    )
                else:
                    most_similar_visits[patient][visit] = sorted(
                        values, key=lambda x: x[1][1], reverse=True
                    )

        return most_similar_visits

    def get_all_visits_ranked(similarities, metric="mse"):
        similarities_ranked = {
            patient: {visit: [] for visit in range(len(similarities[patient]))}
            for patient in similarities
        }

        for patient in tqdm(similarities):
            for visit in similarities[patient]:
                # preprocessing to easily order
                values = similarities[patient][visit].items()
                values = [
                    [(elem[0], i, elem[1][i]) for i in range(len(elem[1]))]
                    for elem in values
                ]
                values = [item for sublist in values for item in sublist]  # flatten
                # sort
                if metric == "mse":
                    similarities_ranked[patient][visit] = sorted(
                        values, key=lambda x: x[2]
                    )
                else:
                    similarities_ranked[patient][visit] = sorted(
                        values, key=lambda x: x[2], reverse=True
                    )
        return similarities_ranked

    mean_similarities_mse, visit_similarities_mse = aggregate_similarities(
        similarities_mse
    )
    mean_similarities_cos, visit_similarities_cos = aggregate_similarities(
        similarities_cos, metric="cos"
    )
    most_similar_patients_mse = find_most_similar_patients(mean_similarities_mse)
    most_similar_patients_cos = find_most_similar_patients(
        mean_similarities_cos, metric="cos"
    )
    most_similar_visits_mse = find_most_similar_visits(visit_similarities_mse)
    most_similar_visits_cos = find_most_similar_visits(
        visit_similarities_cos, metric="cos"
    )
    similarities_ranked_mse = get_all_visits_ranked(similarities_mse)
    similarities_ranked_cos = get_all_visits_ranked(similarities_cos, metric="cos")
    # # cluster normalized data directly
    kmeans_raw = KMeans(n_clusters=k, random_state=seed).fit(raw_histories)
    kmeans_raw.predict(raw_histories_test)

    print("End of script")
