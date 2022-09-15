from scqm.custom_library.cv.cv import CV
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.utils import set_seeds
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
import itertools
import random
import time
import os
import torch
import pickle
import gc


class CVMultitask(CV):
    def __init__(self, dataset, k: int = 5, split_indices: dict = {}) -> None:
        """Init method. Create the partition on the data.

        Args:
            dataset (Dataset): dataset
            k (int): Number of folds
        """
        set_seeds(0)
        self.dataset = dataset
        self.k = k
        self.partition = MultitaskPartition(
            self.dataset, k=self.k, split_indices=split_indices
        )

    def perform_cv(self, parameters: dict, fold: int, n_epochs: int = 40) -> None:
        """Perform a CV on a given fold, and save trained model.

        Args:
            parameters (dict): name of parameters as keys, list of values to try out as values
            fold (int): Fold on which to perform cv.
            n_epochs (int, optional): Number of training epochs. Defaults to 40.

        """
        combinations = list(itertools.product(*parameters.values()))
        self.partition.set_current_fold(fold)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        for ind, (
            num_layers_enc,
            hidden_enc,
            num_layers,
            num_layers_pred,
            hidden_pred,
            lr,
            p,
        ) in enumerate(combinations):
            # # to save
            path = "/cluster/home/ctrottet/runs/scqm/" + time.strftime("%Y%m%d-%H%M")
            os.mkdir(path)

            print(f"{ind} combination out of {len(combinations)}")
            print(
                f"num_layers_enc {num_layers_enc}, hidden_enc {hidden_enc}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, dropout {p}, lr {lr}"
            )
            model_specifics = {
                "num_layers_enc": num_layers_enc,
                "hidden_enc": hidden_enc,
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
            self.dataset.min_num_targets = 2
            trainer = MultitaskTrainer(
                model,
                self.dataset,
                n_epochs,
                batch_size={
                    "das28": int(len(self.dataset) / 15),
                    "basdai": int(len(self.dataset) / (15 * 3)),
                },
                lr=lr,
                balance_classes=True,
                use_early_stopping=False,
            )
            trainer.train_model(model, self.partition, debug_patient=False)
            # for memory
            delattr(trainer, "dataset")
            with open(path + "/params.pkl", "wb") as f:
                pickle.dump(model_specifics, f)
            with open(path + "/trainer.pkl", "wb") as f:
                pickle.dump(trainer, f)
