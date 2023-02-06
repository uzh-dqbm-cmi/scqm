import torch
from scqm.custom_library.data_objects.dataset import Dataset


def apply_attention_analysis(
    model,
    dataset: Dataset,
    patients: list,
    target_name: str,
):
    glob_att = []
    pred_ids = []
    for patient_id in patients:
        with torch.no_grad():
            # method to directly apply the model to a single patient
            if target_name == "das283bsr_score":
                mapping = dataset.mapping_for_masks_das28
                masks = dataset.masks_das28
                target_index_in_events = dataset.event_names.index("a_visit")
                target_index_in_tensor = dataset.target_value_index_das28
                target_tensor = dataset[patient_id].targets_das28_df_tensor
                time_index = dataset.time_index_das28
                targets_df = dataset[patient_id].targets_das28_df
            elif target_name == "asdas_score":
                mapping = dataset.mapping_for_masks_asdas
                masks = dataset.masks_asdas
                target_index_in_events = dataset.event_names.index("a_visit")
                target_index_in_tensor = dataset.target_value_index_asdas
                target_tensor = dataset[patient_id].targets_asdas_df_tensor
                time_index = dataset.time_index_asdas
                targets_df = dataset[patient_id].targets_asdas_df
            patient_mask_index = mapping[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = model.encoders[event](
                    getattr(dataset[patient_id], event + "_df_tensor").to(model.device)
                )

            seq_lengths = masks.seq_lengths[:, patient_mask_index, :]
            available_target_mask = masks.available_target_mask[patient_mask_index]
            predictions = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1),
                device=model.device,
            )
            max_num_visits = masks.num_targets[patient_mask_index]
            total_num = masks.total_num[patient_mask_index]

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1),
                device=model.device,
            )
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1),
                dtype=torch.int64,
                device=model.device,
            )
            time_to_targets = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1, 1),
                device=model.device,
            )
            visit_ids = []
            all_timelines = []
            all_events = {
                index: []
                for index in range(torch.sum(available_target_mask == True).item())
            }
            index_target = 0
            all_attention = {
                index: {event: [] for event in dataset.event_names}
                for index in range(torch.sum(available_target_mask == True).item())
            }
            all_global_attention = []
            # event_count = {event: 0 for event in dataset.event_names}
            events = torch.zeros(
                size=(
                    torch.sum(available_target_mask == True).item(),
                    len(dataset.event_names),
                )
            )
            prediction_ids = []
            for visit in range(0, max_num_visits - dataset.min_num_targets + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_target_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v

                combined = {
                    event: torch.zeros(
                        size=(
                            1,
                            seq_lengths[visit][index].item(),
                            model.config[event]["size_out"],
                        ),
                        device=model.device,
                    )
                    for index, event in enumerate(dataset.event_names)
                }

                combined_lstm = torch.zeros(
                    size=(1, model.combined_history_size[0]), device=model.device
                )
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index] > 0:

                        combined[event][
                            : seq_lengths[visit][index], :
                        ] = encoder_outputs[event][: seq_lengths[visit][index]]

                    else:
                        continue
                _, cropped_ext, cropped_timeline, _, _, _ = dataset[
                    patient_id
                ].get_cropped_timeline(
                    visit + dataset.min_num_targets,
                    masks.min_time_since_last_event,
                    masks.max_time_since_last_event,
                    target_name,
                )
                all_timelines.append(cropped_timeline)
                all_events[index_target] = cropped_ext
                # targets (values)
                target_values[index_target] = target_tensor[
                    visit + dataset.min_num_targets - 1, target_index_in_tensor
                ]
                # TODO change

                time_to_targets[index_target] = target_tensor[
                    visit + dataset.min_num_targets - 1, time_index
                ]
                prediction_ids.append(
                    targets_df["uid_num"].iloc[visit + dataset.min_num_targets - 1]
                )
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index].item() > 0:
                        events[index_target, index] = 1
                        lengths = seq_lengths[visit][index].reshape(1).cpu()
                        pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                            combined[event],
                            batch_first=model.batch_first,
                            lengths=lengths,
                            enforce_sorted=False,
                        )
                        output, (hn, cn) = model.lstm_modules[event](
                            pack_padded_sequence
                        )
                        unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
                            output, batch_first=model.batch_first
                        )[0]
                        attention_weights = torch.nn.Softmax(dim=1)(
                            torch.matmul(unpacked_output, model.Attention[event])
                        )
                        all_attention[index_target][event] = attention_weights
                        combined_lstm[
                            0,
                            model.combined_history_size[1][
                                index
                            ] : model.combined_history_size[1][index + 1],
                        ] = torch.sum(unpacked_output * attention_weights, dim=1)

                combined_lstm_input = torch.reshape(
                    combined_lstm[0],
                    shape=(
                        1,
                        len(dataset.event_names),
                        model.history_size,
                    ),
                )

                general_info = (
                    model.p_encoder(
                        dataset[patient_id].patients_df_tensor.to(model.device)
                    )
                    .reshape(1, 1, model.history_size)
                    .to(model.device)
                )
                combined_input = torch.cat((general_info, combined_lstm_input), dim=1)
                global_attention_weights = torch.nn.Softmax(dim=1)(
                    torch.matmul(combined_input, model.GlobalAttention)
                )
                all_global_attention.append(global_attention_weights)

                combined_input = torch.sum(
                    global_attention_weights * combined_input,
                    dim=1,
                )

                pred_input = torch.cat(
                    (
                        combined_input,
                        time_to_targets[index_target],
                    ),
                    dim=1,
                )
                if target_name == "das283bsr_score":
                    predictions[index_target] = model.PModuleDas28(pred_input)
                else:
                    predictions[index_target] = model.PModuleAsdas(pred_input)
                index_target += 1
            if model.task == "classification":
                predictions = torch.tensor(
                    [torch.argmax(elem) for elem in predictions], device=model.device
                )

        glob_att.extend(all_global_attention)
        pred_ids.extend(prediction_ids)
    return (glob_att, pred_ids)
