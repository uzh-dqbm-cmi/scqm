import itertools
import random
import time
import os
import torch
import pickle
import gc

from scqm.custom_library.cv.cv import CV
from scqm.custom_library.models.adaptive_net import Adaptivenet
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer


class CVAdaptivenet(CV):
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
