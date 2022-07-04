import sys
import pickle
import numpy as np

from scqm.custom_library.cv.adaptive_net import CVAdaptivenet

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
