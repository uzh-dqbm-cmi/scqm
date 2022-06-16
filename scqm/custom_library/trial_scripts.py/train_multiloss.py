import torch
import pickle

import io

from scqm.custom_library.models.other_net_multiloss import OthernetMultiloss
from scqm.custom_library.trainers.multiloss import MultilossTrainer


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


if __name__ == "__main__":
    with open("/opt/tmp/saved_cv_ada.pickle", "rb") as handle:
        # dataset = pickle.load(handle)
        cv = CPU_Unpickler(handle).load()
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

    model = OthernetMultiloss(model_specifics, device)

    batch_size = int(len(dataset) / 15)
    trainer = MultilossTrainer(
        model,
        dataset,
        n_epochs=60,
        batch_size=batch_size,
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )
    trainer.train_model(model, partition, debug_patient=False)

    delattr(trainer, "dataset")
    with open("/opt/tmp/trainer_multiloss_more_epochs.pickle", "wb") as handle:
        pickle.dump(trainer, handle)
