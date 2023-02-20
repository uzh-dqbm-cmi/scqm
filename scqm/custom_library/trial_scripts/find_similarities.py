from scqm.custom_library.utils import set_seeds
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing
from scqm.custom_library.preprocessing.select_features import extract_adanet_features
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv.adaptive_net import CVAdaptivenet
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.results.multiclass_results import MulticlassResults
from scqm.custom_library.results.baseline_results import BaselineResults
from scqm.custom_library.results.utils import get_naive_baseline
from scqm.custom_library.plot.shap.shap_computer import ShapComputer
from scqm.custom_library.clustering.cluster_analysis import ClusterAnalysis

import pandas as pd
import numpy as np
from scqm.custom_library.post_hoc.processings import drugs_taken_at_prediction
from scqm.custom_library.post_hoc.plots import plot_average_days_to_visit
import pickle
from tqdm import tqdm
from scqm.custom_library.results.multiclass_results import MulticlassResults
from sklearn.neighbors import NearestNeighbors
from scqm.custom_library.clustering.knn_similarity import get_knn_similarity_performance


if __name__ == "__main__":
    set_seeds(0)
    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_10_11.pickle", "rb"
    ) as f:
        cv = pickle.load(f)
    with open(
        "/cluster/work/medinfmk/scqm/tmp/cluster_analysis_all.pickle", "rb"
    ) as handle:
        cluster_analysis = pickle.load(handle)
    dataset = cv.dataset
    subset = dataset.test_ids
    subset_das28 = [
        p for p in subset if dataset[p].target_name in ["both", "das283bsr_score"]
    ]
    subset_asdas = [
        p for p in subset if dataset[p].target_name in ["both", "asdas_score"]
    ]
    subset_train = dataset.train_ids
    subset_das28_train = [
        p for p in subset_train if dataset[p].target_name in ["both", "das283bsr_score"]
    ]
    subset_asdas_train = [
        p for p in subset_train if dataset[p].target_name in ["both", "asdas_score"]
    ]
    trainers = {}
    results = {}
    results_train = {}
    models = {}
    df_das28_dict, df_asdas_dict, metrics_das28, metrics_asdas = {}, {}, {}, {}
    (
        df_das28_train_dict,
        df_asdas_train_dict,
        metrics_train_das28,
        metrics_train_asdas,
    ) = ({}, {}, {}, {})
    print("Computing clusters")
    for fold in range(0, 1):
        print(fold)
        with open(
            "/cluster/work/medinfmk/scqm/tmp/fold"
            + str(fold)
            + "/10_11_2022_att/trainer_"
            + str(fold)
            + "_5.pickle",
            "rb",
        ) as handle:
            trainers[fold] = pickle.load(handle)
            print(trainers[fold].optimal_epoch)
            models[fold] = trainers[fold].best_model
            results[fold] = MulticlassResults(dataset, models[fold], trainers[fold])
            (
                df_das28_dict[fold],
                df_asdas_dict[fold],
                metrics_das28[fold],
                metrics_asdas[fold],
            ) = results[fold].evaluate_model(dataset.test_ids)
            results_train[fold] = MulticlassResults(
                dataset, models[fold], trainers[fold]
            )
            (
                df_das28_train_dict[fold],
                df_asdas_train_dict[fold],
                metrics_train_das28[fold],
                metrics_train_asdas[fold],
            ) = results_train[fold].evaluate_model(dataset.train_ids)
            # cluster_analysis[fold] = ClusterAnalysis(dataset, models[fold], trainers[fold], n_clusters=2)

    df_das28_drugs = {}
    df_das28_drugs_exp = {}
    df_asdas_drugs = {}
    df_asdas_drugs_exp = {}
    df_das28_drugs_train = {}
    df_das28_drugs_exp_train = {}
    df_asdas_drugs_train = {}
    df_asdas_drugs_exp_train = {}
    print("computing drug exp")
    for key in df_das28_dict:
        df_das28_drugs[key], df_das28_drugs_exp[key] = drugs_taken_at_prediction(
            dataset, df_das28_dict[key], subset_das28, "das283bsr_score"
        )
        df_asdas_drugs[key], df_asdas_drugs_exp[key] = drugs_taken_at_prediction(
            dataset, df_asdas_dict[key], subset_asdas, "asdas_score"
        )

        (
            df_das28_drugs_train[key],
            df_das28_drugs_exp_train[key],
        ) = drugs_taken_at_prediction(
            dataset, df_das28_train_dict[key], subset_das28_train, "das283bsr_score"
        )
        (
            df_asdas_drugs_train[key],
            df_asdas_drugs_exp_train[key],
        ) = drugs_taken_at_prediction(
            dataset, df_asdas_train_dict[key], subset_asdas_train, "asdas_score"
        )

    #K = 50

    K_asdas = [1, 5, 10, 20, 30, 50, 250, 750, 1500, 2500, 3500, 4500, 5000]
    K_das28 = [1, 5, 10, 20, 30, 50, 500, 2000, 10000, 15000, 20000, 25000, 30000]
    similarities_das28 = np.empty((len(K_das28), 5))
    similarities_asdas = np.empty((len(K_asdas), 5))

    for fold in range(1):
        index = 0
        print(fold)
        for k in K_asdas:
            print(k)
            # (
            #     das28_similar_targets_mse_knn,
            #     das28_baseline_targets_mse_knn,
            #     das28_random_targets_mse_knn,
            #     _,
            #     _,
            #     _,
            # ) = get_knn_similarity_performance(
            #     df_das28_drugs[fold].copy(),
            #     df_das28_drugs_train[fold].copy(),
            #     cluster_analysis[fold],
            #     tol=1,
            #     k=k,
            # )
            (
                asdas_similar_targets_mse_knn,
                asdas_baseline_targets_mse_knn,
                asdas_random_targets_mse_knn,
                _,
                _,
                _,
            ) = get_knn_similarity_performance(
                df_asdas_drugs[fold].copy(),
                df_asdas_drugs_train[fold].copy(),
                cluster_analysis[fold],
                tol=1,
                k=k,
            )
            # similarities_das28[index, fold] = das28_similar_targets_mse_knn
            similarities_asdas[index, fold] = asdas_similar_targets_mse_knn
            index += 1

    for fold in range(1):
        index = 0
        print(fold)
        for k in K_das28:
            print(k)
            (
                das28_similar_targets_mse_knn,
                das28_baseline_targets_mse_knn,
                das28_random_targets_mse_knn,
                _,
                _,
                _,
            ) = get_knn_similarity_performance(
                df_das28_drugs[fold].copy(),
                df_das28_drugs_train[fold].copy(),
                cluster_analysis[fold],
                tol=1,
                k=k,
            )
            # (
            #     asdas_similar_targets_mse_knn,
            #     asdas_baseline_targets_mse_knn,
            #     asdas_random_targets_mse_knn,
            #     _,
            #     _,
            #     _,
            # ) = get_knn_similarity_performance(
            #     df_asdas_drugs[fold].copy(),
            #     df_asdas_drugs_train[fold].copy(),
            #     cluster_analysis[fold],
            #     tol=1,
            #     k=k,
            # )
            similarities_das28[index, fold] = das28_similar_targets_mse_knn
            # similarities_asdas[index, fold] = asdas_similar_targets_mse_knn
            index += 1
    print(similarities_das28)
    print(similarities_asdas)
    with open("/cluster/work/medinfmk/scqm/tmp/simil_das28_incr_k.pickle", "wb") as handle:
        pickle.dump(similarities_das28, handle)

    with open("/cluster/work/medinfmk/scqm/tmp/simil_asdas_incr_k.pickle", "wb") as handle:
        pickle.dump(similarities_asdas, handle)
