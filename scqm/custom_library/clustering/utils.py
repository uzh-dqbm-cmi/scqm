import torch
from scqm.custom_library.data_objects.dataset import Dataset


def get_features(
    model,
    dataset: Dataset,
    patient_id: str,
    target_name: str,
):
    with torch.no_grad():
        # method to directly apply the model to a single patient
        if target_name not in dataset[patient_id].targets.keys():
            raise ValueError("target name not available for this patient")
        else:
            if target_name == "das283bsr_score":
                mapping = dataset.mapping_for_masks_das28
                masks = dataset.masks_das28

            elif target_name == "basdai_score":
                mapping = dataset.mapping_for_masks_basdai
                masks = dataset.masks_basdai

        patient_mask_index = mapping[patient_id]
        event_tensors = {}

        for event in dataset.event_names:
            event_tensors[event] = getattr(
                dataset[patient_id], event + "_df_tensor"
            ).to(model.device)

        general_info = dataset[patient_id].patients_df_tensor.to(model.device)

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
        for t in range(0, max_num_targets - dataset.min_num_targets + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(available_target_mask[t] == True).item() == 0:
                continue

            for index, event in enumerate(dataset.event_names):
                if seq_lengths[t][index] > 0:

                    combined[index_target][event] = event_tensors[event][
                        seq_lengths[t][index] - 1
                    ]

                else:
                    continue
            combined[index_target]["general"] = general_info
            combined_concat[index_target] = torch.cat(
                [combined[index_target][event] for event in dataset.event_names]
                + [general_info.flatten()]
            )
            index_target += 1
            # pred_input = torch.cat(
            #     (
            #         general_info,
            #         combined_lstm_input,
            #         time_to_targets[index_target],
            #     ),
            #     dim=1,
            # )
        return combined, combined_concat
