import numpy as np
import torch
from scqm.custom_library.data_objects.dataset import Dataset
import matplotlib.pyplot as plt


def plot_colormap(df, xlabels, xlabel, ylabels, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(np.array(df, dtype=np.float32))
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return


def find_drugs(meds_list, patient, dataset):
    med_df = dataset[patient].med_df.copy()
    med_names = []
    for elem in meds_list:
        if elem[0] == "med_s":
            name = med_df.loc[
                (med_df.med_id == elem[1]) & (med_df.is_start == 1)
            ].medication_generic_drug.item()
        else:
            name = med_df.loc[
                (med_df.med_id == elem[1]) & (med_df.is_start == 0)
            ].medication_generic_drug.item()
        med_names.append((elem[0], name))
    return med_names


def get_all_attention(model, dataset, patients):
    meds_all = {patient: {} for patient in patients}
    meds_and_attention = {patient: {} for patient in patients}
    values = {patient: [] for patient in patients}
    all_patients_all_attention = {patient: {} for patient in patients}
    for patient in patients:
        all_events, all_attention, all_global_attention, events = apply_attention(
            model, dataset, patient
        )
        meds_all[patient] = {
            index: find_drugs(
                [(elem[1], elem[2]) for elem in all_events[index] if "med_" in elem[1]],
                patient,
                dataset,
            )
            for index in range(len(all_events))
        }
        meds_and_attention[patient] = {
            index: [
                meds_all[patient][index][index_med]
                + (all_attention[index]["med"].flatten()[index_med].item(),)
                for index_med in range(len(meds_all[patient][index]))
            ]
            for index in range(len(all_events))
        }
        values[patient] = np.array(
            [
                np.array(
                    [med_event[2] for med_event in meds_and_attention[patient][index]]
                )
                for index in range(len(meds_and_attention[patient]))
            ],
            dtype=object,
        )
        all_patients_all_attention[patient] = all_attention
    return meds_all, meds_and_attention, values, all_patients_all_attention


def apply_attention(model, dataset: Dataset, patient_id: str):
    with torch.no_grad():
        # method to directly apply the model to a single patient
        target_name = model.target_name
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
        patient_mask_index = dataset.mapping_for_masks[patient_id]
        encoder_outputs = {}
        for event in dataset.event_names:
            encoder_outputs[event] = model.encoders[event](
                getattr(dataset[patient_id], event + "_df_tensor").to(model.device)
            )

        seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
        available_target_mask = dataset.masks.available_target_mask[patient_mask_index]
        predictions = torch.empty(
            size=(torch.sum(available_target_mask == True).item(), 1),
            device=model.device,
        )
        max_num_visits = dataset.masks.num_targets[patient_mask_index]
        total_num = dataset.masks.total_num[patient_mask_index]

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

                    combined[event][: seq_lengths[visit][index], :] = encoder_outputs[
                        event
                    ][: seq_lengths[visit][index]]

                else:
                    continue
            _, cropped_ext, cropped_timeline, _, _, _ = dataset[
                patient_id
            ].get_cropped_timeline(
                visit + dataset.min_num_targets,
                dataset.masks.min_time_since_last_event,
                dataset.masks.max_time_since_last_event,
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
                    output, (hn, cn) = model.lstm_modules[event](pack_padded_sequence)
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
            global_attention_weights = torch.nn.Softmax(dim=1)(
                torch.matmul(combined_lstm_input, model.GlobalAttention)
            )
            all_global_attention.append(global_attention_weights)
            combined_lstm_input = torch.reshape(
                global_attention_weights * combined_lstm_input,
                shape=(
                    1,
                    model.combined_history_size[0],
                ),
            )
            # concat computed patient history with general information
            general_info = model.p_encoder(
                dataset[patient_id].patients_df_tensor.to(model.device)
            )
            pred_input = torch.cat(
                (
                    general_info,
                    combined_lstm[available_target_mask[visit]].reshape(
                        1, combined_lstm[available_target_mask[visit]].shape[2]
                    ),
                    time_to_targets[index_target],
                ),
                dim=1,
            )
            predictions[index_target] = model.PModule(pred_input)
            index_target += 1
        if model.task == "classification":
            predictions = torch.tensor(
                [torch.argmax(elem) for elem in predictions], device=model.device
            )

    return all_events, all_attention, all_global_attention, events


def regularise_array(arr, val=-1):
    """Takes irregular array and returns regularised masked array

    This first pads the irregular awway *arr* with values *val* to make
    it of rectangular. It then applies a mask so that the padded values
    are not displayed by pcolormesh. For this reason val should not
    be in *arr* as you will loose these points.
    https://stackoverflow.com/questions/24803090/python-matplotlib-imshow-with-difference-lenghts-in-data-array
    """

    lengths = [len(d) for d in arr]
    max_length = max(lengths)
    reg_array = np.zeros(shape=(arr.shape[0], max_length))

    for i in np.arange(arr.shape[0]):
        reg_array[i] = np.append(arr[i], np.zeros(max_length - lengths[i]) + val)

    reg_array = np.ma.masked_array(reg_array, reg_array == val)

    return reg_array
