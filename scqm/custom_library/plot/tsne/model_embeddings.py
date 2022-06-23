import numpy as np
import torch
from scqm.custom_library.data_objects.dataset import Dataset


def get_history(self, dataset: Dataset, patient_id: str):
    with torch.no_grad():
        # method to directly apply the model to a single patient
        patient_mask_index = dataset.mapping_for_masks[patient_id]
        encoder_outputs = {}
        # basdai or das28
        target_name = dataset[patient_id].target_name
        if target_name == "das283bsr_score":
            target_index_in_events = dataset.event_names.index("a_visit")
            target_index_in_tensor = dataset.target_value_index_das28
            target_tensor = dataset[patient_id].targets_das28_df_tensor

            time_index = dataset.time_index_das28
        else:
            target_index_in_events = dataset.event_names.index("basdai")
            target_index_in_tensor = dataset.target_value_index_basdai
            target_tensor = dataset[patient_id].targets_basdai_df_tensor

            time_index = dataset.time_index_basdai

        for event in dataset.event_names:
            encoder_outputs[event] = self.encoders[event](
                getattr(dataset[patient_id], event + "_df_tensor").to(self.device)
            )

        seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
        available_target_mask = dataset.masks.available_target_mask[patient_mask_index]
        predictions = torch.empty(
            size=(torch.sum(available_target_mask == True).item(), 1),
            device=self.device,
        )
        max_num_targets = dataset.masks.num_targets[patient_mask_index]
        total_num = dataset.masks.total_num[patient_mask_index]
        # to store the input for predictions
        pred_input_all = torch.empty(
            size=(
                torch.sum(available_target_mask == True).item(),
                self.combined_history_size[0] + 1 + 1,
            )
        )
        # targets (values)
        target_values = torch.empty(
            size=(torch.sum(available_target_mask == True).item(), 1),
            device=self.device,
        )
        time_to_targets = torch.empty(
            size=(torch.sum(available_target_mask == True).item(), 1, 1),
            device=self.device,
        )

        index_target = 0
        for t in range(0, max_num_targets - dataset.min_num_targets + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(available_target_mask[t] == True).item() == 0:
                continue
            # create combined ordered list of visit/medication/events up to v

            combined = {
                event: torch.zeros(
                    size=(
                        1,
                        seq_lengths[t][index].item(),
                        self.config[event]["size_out"],
                    ),
                    device=self.device,
                )
                for index, event in enumerate(dataset.event_names)
            }

            combined_lstm = torch.zeros(
                size=(1, self.combined_history_size[0]), device=self.device
            )
            for index, event in enumerate(dataset.event_names):
                if seq_lengths[t][index] > 0:

                    combined[event][: seq_lengths[t][index], :] = encoder_outputs[
                        event
                    ][: seq_lengths[t][index]]

                else:
                    continue

            # targets (values)
            target_values[index_target] = target_tensor[
                t + dataset.min_num_targets - 1,
                target_index_in_tensor,
            ]
            # TODO change

            time_to_targets[index_target] = target_tensor[
                t + dataset.min_num_targets - 1, time_index
            ]

            # target_categories[index_target] = dataset[patient_id].targets_df_tensor[
            #     visit + dataset.min_num_targets - 1, dataset.target_index
            # ]
            for index, event in enumerate(dataset.event_names):
                if seq_lengths[t][index].item() > 0:
                    lengths = seq_lengths[t][index].reshape(1).cpu()
                    pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                        combined[event],
                        batch_first=self.batch_first,
                        lengths=lengths,
                        enforce_sorted=False,
                    )
                    output, (hn, cn) = self.lstm_modules[event](pack_padded_sequence)
                    unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
                        output, batch_first=self.batch_first
                    )[0]
                    attention_weights = torch.nn.Softmax(dim=1)(
                        torch.matmul(unpacked_output, self.Attention[event])
                    )
                    combined_lstm[
                        0,
                        self.combined_history_size[1][
                            index
                        ] : self.combined_history_size[1][index + 1],
                    ] = torch.sum(unpacked_output * attention_weights, dim=1)
            combined_lstm_input = torch.reshape(
                combined_lstm[0],
                shape=(
                    1,
                    len(dataset.event_names),
                    self.history_size,
                ),
            )
            global_attention_weights = torch.nn.Softmax(dim=1)(
                torch.matmul(combined_lstm_input, self.GlobalAttention)
            )
            combined_lstm_input = torch.reshape(
                global_attention_weights * combined_lstm_input,
                shape=(
                    1,
                    self.combined_history_size[0],
                ),
            )
            # concat computed patient history with general information
            general_info = self.p_encoder(
                dataset[patient_id].patients_df_tensor.to(self.device)
            )
            pred_input = torch.cat(
                (
                    general_info,
                    combined_lstm_input,
                    time_to_targets[index_target],
                ),
                dim=1,
            )
            pred_input_all[index_target] = pred_input
            if target_name == "das283bsr_score":
                predictions[index_target] = self.PModuleDas28(pred_input)
            else:
                predictions[index_target] = self.PModuleBasdai(pred_input)
            index_target += 1

    return (predictions, target_values, time_to_targets, pred_input_all)


def get_embeddings(model, dataset, subset=None):
    if subset is None:
        subset = dataset.test_ids
    indices = []
    for p in subset:
        indices.append(dataset.mapping_for_masks[p])
    seq_lengths = dataset.masks.seq_lengths[:, indices, :]
    num_targets = [
        torch.sum(dataset.masks.available_target_mask[elem] == True).item()
        for elem in indices
    ]
    embeddings = torch.empty(
        size=(np.array(num_targets).sum(), model.combined_history_size[0] + 1 + 1)
    )
    index = 0
    patient_in_embedding = {
        patient: {"indices": [], "target": dataset[patient].target_name}
        for i, patient in enumerate(subset)
    }
    target_values = torch.empty(size=(np.array(num_targets).sum(), 1))
    for i, patient in enumerate(subset):
        (
            _,
            target_values[index : index + num_targets[i]],
            _,
            embeddings[index : index + num_targets[i], :],
        ) = get_history(model, dataset, patient)
        patient_in_embedding[patient]["indices"] = np.arange(
            index, index + num_targets[i]
        )
        index += num_targets[i]

    print(index)
    return embeddings, patient_in_embedding, target_values
