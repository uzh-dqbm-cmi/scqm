from scqm.custom_library.trainers.mlp import MLPTrainer
from scqm.custom_library.models.MLP import MLP
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing
from scqm.custom_library.preprocessing.select_features import extract_adanet_features
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv.adaptive_net import CVAdaptivenet
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.utils import set_seeds
from scqm.custom_library.results.multiclass_results import MulticlassResults
from scqm.custom_library.results.results import Results
import pandas as pd

import numpy as np
import random
import torch
import pickle
import itertools
from tqdm import tqdm

if __name__ == "__main__":
    set_seeds(0)
    print("train mlp")
    target_name = "asdas_score"

    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_16_09.pickle", "rb"
    ) as f:
        cv = pickle.load(f)
    dataset = cv.dataset
    partition = cv.partition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if target_name == "das283bsr_score":
        input_size = dataset.joint_das28_df_scaled_tensor_train.shape[1]
        config = {
            "input_size": input_size,
            "output_size": 1,
            "num_hidden": 5,
            "hidden_size": 100,
        }
    elif target_name == "asdas_score":
        input_size = dataset.joint_asdas_df_scaled_tensor_train.shape[1]
        config = {
            "input_size": input_size,
            "output_size": 1,
            "num_hidden": 5,
            "hidden_size": 100,
        }
        # 40
    for fold in range(5):

        partition.set_current_fold(fold)
        model = MLP(config, device)
        trainer = MLPTrainer(
            model,
            dataset,
            n_epochs=400,
            batch_size=300,
            lr=1e-3,
            balance_classes=True,
            use_early_stopping=False,
            target_name=target_name,
        )
        trainer.train_model(model, partition, verbose=False)
        delattr(trainer, "dataset")
        # with open(
        #     "/cluster/work/medinfmk/scqm/tmp/baselines/mlp_trainer_asdas_" + str(fold) + ".pickle",
        #     "wb",
        # ) as handle:
        #     pickle.dump(trainer, handle)
