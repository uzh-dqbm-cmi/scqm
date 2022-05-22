import numpy as np
import random
import torch
import pickle
import gc
import io
import copy

from scqm.custom_library.models.adaptive_net import Adaptivenet
from scqm.custom_library.models.adaptive_net_with_attention import AdaptivenetWithAttention
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.preprocessing.preprocessing import (
    load_dfs_all_data,
    preprocessing,
    extract_other_features,
    extract_adanet_features,
)
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv import CVAdaptivenet
from scqm.custom_library.results.results import Results
from scqm.custom_library.utils import set_seeds


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


if __name__ == "__main__":

    set_seeds(0)
    np.random.get_state()[1][0]

    with open("/opt/tmp/saved_cv_ada.pickle", "rb") as handle:
        cv = pickle.load(handle)
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
    model_specifics = {
        "task": "regression",
        "num_targets": 1,
        "size_embedding": 5,
        "num_layers_enc": 2,
        "hidden_enc": 100,
        "size_history": 20,
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
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )
    # model = AdaptivenetWithAttention(model_specifics, device)
    model = Adaptivenet(model_specifics, device)
    # len(partition.partitions_train[partition.current_fold])
    batch_size = int(len(dataset) / 15)
    trainer = AdaptivenetTrainer(
        model,
        dataset,
        n_epochs=50,
        batch_size=batch_size,
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )
    print(f"batch size {batch_size}")
    trainer.train_model(model, partition, debug_patient=False)
    delattr(trainer, "dataset")
    with open("/opt/tmp/ada_1805.pickle", "wb") as handle:
        pickle.dump(trainer, handle)
