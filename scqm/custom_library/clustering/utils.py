import torch
from scqm.custom_library.data_objects.dataset import Dataset
from tqdm import tqdm
import numpy as np


def get_features(
    model,
    dataset: Dataset,
    patient_id: str,
    target_name: str,
):
    with torch.no_grad():

        if target_name not in dataset[patient_id].targets_df.keys():
            raise ValueError("target name not available for this patient")
        else:
            if target_name == "das283bsr_score":
                mapping = dataset.mapping_for_masks_das28
                masks = dataset.masks_das28

            elif target_name == "asdas_score":
                mapping = dataset.mapping_for_masks_asdas
                masks = dataset.masks_asdas

        patient_mask_index = mapping[patient_id]
        event_tensors = {}
        event_unscaled = {}
        for event in dataset.event_names:
            event_tensors[event] = getattr(
                dataset[patient_id], event + "_df_tensor"
            ).to(model.device)
            event_unscaled[event] = getattr(dataset[patient_id], event + "_df").values

        general_info = dataset[patient_id].patients_df_tensor.to(model.device)
        general_info_unscaled = dataset[patient_id].patients_df.values

        seq_lengths = masks.seq_lengths[:, patient_mask_index, :]
        available_target_mask = masks.available_target_mask[patient_mask_index]

        max_num_targets = masks.num_targets[patient_mask_index]

        combined = {
            t: {
                event: torch.zeros(
                    size=(model.config[event]["num_features"],),
                    device=model.device,
                )
                for index, event in enumerate(dataset.event_names)
            }
            for t in range(torch.sum(available_target_mask == True).item())
        }
        combined_unscaled = {
            t: {
                event: torch.full(
                    size=(event_unscaled[event].shape[1],),
                    device="cpu",
                    fill_value=np.nan,
                )
                for index, event in enumerate(dataset.event_names)
            }
            for t in range(torch.sum(available_target_mask == True).item())
        }
        combined_unscaled_all = {
            t: {event: {} for index, event in enumerate(dataset.event_names)}
            for t in range(torch.sum(available_target_mask == True).item())
        }
        index_target = 0
        combined_concat = torch.zeros(
            size=(
                torch.sum(available_target_mask == True).item(),
                sum(
                    [
                        model.config[event]["num_features"]
                        for event in dataset.event_names
                    ]
                )
                + model.config["num_general_features"],
            )
        )
        combined_concat_unscaled = np.zeros(
            shape=(
                torch.sum(available_target_mask == True).item(),
                sum([event_unscaled[event].shape[1] for event in dataset.event_names])
                + general_info_unscaled.shape[1],
            ),
            dtype="object",
        )
        for t in range(0, max_num_targets - dataset.min_num_targets + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(available_target_mask[t] == True).item() == 0:
                continue

            for index, event in enumerate(dataset.event_names):
                if seq_lengths[t][index] > 0:

                    combined[index_target][event] = event_tensors[event][
                        seq_lengths[t][index] - 1
                    ]
                    combined_unscaled[index_target][event] = event_unscaled[event][
                        seq_lengths[t][index] - 1
                    ]
                    combined_unscaled_all[index_target][event] = event_unscaled[event][
                        : seq_lengths[t][index]
                    ]

                else:
                    continue

            combined[index_target]["general"] = general_info
            combined_unscaled[index_target]["general"] = general_info_unscaled
            combined_concat[index_target] = torch.cat(
                [combined[index_target][event] for event in dataset.event_names]
                + [general_info.flatten()]
            )
            combined_concat_unscaled[index_target] = np.concatenate(
                [
                    combined_unscaled[index_target][event]
                    for event in dataset.event_names
                ]
                + [general_info_unscaled.flatten()]
            )
            index_target += 1

        return (
            combined,
            combined_concat,
            combined_unscaled,
            combined_concat_unscaled,
            combined_unscaled_all,
        )


def get_histories_and_features(dataset, model, subset):

    subset_das28 = [
        p for p in subset if dataset[p].target_name in ["both", "das283bsr_score"]
    ]
    subset_asdas = [p for p in subset if dataset[p].target_name == "asdas_score"]

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
                dataset.masks_asdas.available_target_mask[
                    dataset.mapping_for_masks_asdas[patient]
                ]
                == True
            ).item()
            for patient in subset_asdas
        ]
    )
    model_histories = torch.empty(size=(sum(numbers_of_target), model.pred_input_size))
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
    patient_in_embedding = {
        patient: {"indices": [], "target": dataset[patient].target_name}
        for patient in subset_das28 + subset_asdas
    }
    index_in_history = 0
    raw_features = {}
    raw_features_all = {}
    raw_features_unscaled = {}
    print(f"saving histories for das28")
    hist_per_event_all = {}
    for index, patient in enumerate(tqdm(subset_das28)):
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
            raw_features_all[patient],
        ) = get_features(model, dataset, patient, "das283bsr_score")
        model_histories[
            index_in_history : index_in_history + numbers_of_target[index]
        ] = hist
        hist_per_event_all[patient] = hist_per_event
        patient_in_embedding[patient]["indices"] = np.arange(
            index_in_history, index_in_history + numbers_of_target[index]
        )
        index_in_history += numbers_of_target[index]

    print(f"saving histories for asdas")

    for index, patient in enumerate(tqdm(subset_asdas)):
        _, _, _, hist, hist_per_event, _, _ = model.apply(
            dataset, patient, "asdas_score", return_history=True
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
            raw_features_all[patient],
        ) = get_features(model, dataset, patient, "asdas_score")
        model_histories[
            index_in_history : index_in_history
            + numbers_of_target[index + len(subset_das28)]
        ] = hist
        hist_per_event_all[patient] = hist_per_event
        patient_in_embedding[patient]["indices"] = np.arange(
            index_in_history,
            index_in_history + numbers_of_target[index + len(subset_das28)],
        )
        index_in_history += numbers_of_target[index + len(subset_das28)]

    return (
        raw_features,
        raw_histories,
        raw_features_all,
        raw_features_unscaled,
        raw_histories_unscaled,
        model_histories,
        subset_das28,
        subset_asdas,
        patient_in_embedding,
        hist_per_event_all,
    )
