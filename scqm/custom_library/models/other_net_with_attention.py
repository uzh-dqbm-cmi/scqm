import torch
import numpy as np

from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import EventEncoder

from scqm.custom_library.models.modules.lstms import LstmEventSpecific
from scqm.custom_library.models.modules.predictions import PredModule


from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch


class OthernetOptimizedWithAttention(Model):
    def __init__(self, config: dict, device: str):
        super().__init__(config, device)
        self.task = "regression"
        self.size_embedding = config["size_embedding"]
        # tuple storing the sum of the history sizes (= input size for prediction module)
        # and the cumulative sum of the history sizes, used as indices later to store the
        # outputs of the individual lstms in a combined vector.
        self.combined_history_size = (
            sum([config[name]["size_history"] for name in config["event_names"]]),
            np.cumsum(
                [0] + [config[name]["size_history"] for name in config["event_names"]]
            ),
        )
        self.config = config

        self.encoders = {
            name: EventEncoder(
                config[name]["num_features"],
                config[name]["size_out"],
                config["num_layers_enc"],
                config["hidden_enc"],
                config["dropout"],
            ).to(device)
            for name in config["event_names"]
        }

        self.p_encoder = EventEncoder(
            config["patients"]["num_features"],
            config["patients"]["size_out"],
            config["num_layers_enc"],
            config["hidden_enc"],
            config["dropout"],
        ).to(device)

        self.lstm_modules = {
            name: LstmEventSpecific(
                config[name]["size_out"],
                config["device"],
                config["batch_first"],
                config[name]["size_history"],
                config["num_layers"],
            ).to(device)
            for name in config["event_names"]
        }
        self.Attention = {
            name: torch.nn.Parameter(
                torch.ones(size=(config[name]["size_history"], 1), device=self.device),
                requires_grad=True,
            )
            for name in config["event_names"]
        }
        # + 1 for time to prediction
        self.PModule = PredModule(
            self.combined_history_size[0] + config["patients"]["size_out"] + 1,
            1,
            config["num_layers_pred"],
            config["hidden_pred"],
            config["dropout"],
        ).to(device)
        self.batch_first = config["batch_first"]

        self.parameters = list(self.PModule.parameters()) + list(
            self.p_encoder.parameters()
        )
        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())
            self.parameters += list(self.lstm_modules[name].parameters())
            self.parameters += [self.Attention[name]]

    def train(self):
        for name in self.encoders:
            self.encoders[name].train()
            self.lstm_modules[name].train()
        self.p_encoder.train()
        self.PModule.train()

    def eval(self):

        for name in self.encoders:
            self.encoders[name].eval()
            self.lstm_modules[name].eval()
        self.p_encoder.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset: Dataset, criterion: torch.nn, batch: Batch):
        loss = 0
        encoder_outputs = {}
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, "indices_" + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + "_df_scaled_tensor_train")[indices]
                )
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients]
        )
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            # sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros(
                (len(dataset.event_names),), dtype=torch.int32, device=self.device
            )
            indices_lstm = torch.zeros(
                (len(dataset.event_names),), dtype=torch.int32, device=self.device
            )
            visit_index = dataset.event_names.index("a_visit")

            index_target = 0
            debug_index_target = None
            # to contain encoder outputs. For each event, each patient is represented by a vector of shape max_num_events * embedding size
            # if a patient has less events, the last rows are zero padded
            combined = {
                event: torch.zeros(
                    size=(
                        len(batch.current_indices),
                        batch.seq_lengths[v][:, index].max().item(),
                        self.config[event]["size_out"],
                    ),
                    device=self.device,
                )
                for index, event in enumerate(dataset.event_names)
            }

            # to slice encoder outputs
            indices_to_keep = {event: [] for event in dataset.event_names}
            indices_combined = {event: [] for event in dataset.event_names}
            indices_targets = [
                batch.seq_lengths[v][p, visit_index]
                + batch.total_num[:p, visit_index].sum()
                for p in range(len(batch.seq_lengths[v]))
            ]
            for index, event in enumerate(dataset.event_names):
                indices_to_keep[event].extend(
                    np.arange(0, batch.seq_lengths[v, 0, index], 1)
                )
                indices_combined[event].extend(
                    np.arange(0, batch.seq_lengths[v, 0, index], 1)
                )

                for i in range(1, len(batch.seq_lengths[v])):
                    indices_to_keep[event].extend(
                        np.arange(
                            batch.total_num[:i, index].sum(),
                            batch.total_num[:i, index].sum()
                            + batch.seq_lengths[v, i, index],
                            1,
                        )
                    )
                    indices_combined[event].extend(
                        np.arange(
                            batch.seq_lengths[v][:, index].max().item() * i,
                            batch.seq_lengths[v][:, index].max().item() * i
                            + batch.seq_lengths[v, i, index],
                            1,
                        )
                    )

                #                 print(indices_to_keep[event])
                #                 print(indices_combined[event])
                #                 print(combined[event].shape)
                #                 print(combined[event])
                #                 print(encoder_outputs[event].shape)
                #                 print(encoder_outputs[event])
                if len(indices_to_keep[event]) > 0:
                    combined[event].view(
                        (
                            combined[event].shape[0] * combined[event].shape[1],
                            combined[event].shape[2],
                        )
                    )[indices_combined[event]] = encoder_outputs[event][
                        indices_to_keep[event]
                    ]
            # targets
            target_values = dataset.targets_df_scaled_tensor_train[
                batch.indices_targets
            ][indices_targets, dataset.target_value_index]
            target_values = target_values.reshape(len(target_values), 1)[
                batch.available_visit_mask[:, v]
            ]
            time_to_targets = dataset.targets_df_scaled_tensor_train[
                batch.indices_targets
            ][indices_targets, dataset.time_index]
            time_to_targets = time_to_targets.reshape(len(time_to_targets), 1)[
                batch.available_visit_mask[:, v]
            ]
            target_categories = dataset.targets_df_scaled_tensor_train[
                batch.indices_targets
            ][indices_targets, dataset.target_index]
            target_categories = target_categories.reshape(len(target_categories), 1)[
                batch.available_visit_mask[:, v]
            ]
            # lengths = {event: batch.seq_lengths[v,index, :] for index, event in enumerate(dataset.event_names)}
            # combined representation (before prediction)
            combined_lstm = torch.zeros(
                size=(len(batch.current_indices), self.combined_history_size[0]),
                device=self.device,
            )

            for index, event in enumerate(dataset.event_names):
                # patients with visit to predict and available event
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                patients = [
                    elem.item()
                    for elem in patients
                    if batch.available_visit_mask[elem, v]
                ]
                if len(patients) > 0:
                    # compute the lengths of the sequences for each patient with available visit v
                    lengths = batch.seq_lengths[v, patients, index].cpu()

                    pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                        combined[event][patients, :, :],
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
                        patients,
                        self.combined_history_size[1][
                            index
                        ] : self.combined_history_size[1][index + 1],
                    ] = torch.sum(unpacked_output * attention_weights, dim=1)
            general_info = patient_encoding[batch.available_visit_mask[:, v]]

            pred_input = torch.cat(
                (
                    general_info,
                    combined_lstm[batch.available_visit_mask[:, v]],
                    time_to_targets,
                ),
                dim=1,
            )

            # apply prediction module
            out = self.PModule(pred_input)

            targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)

            if debug_index_target != None:
                # print(out)
                print(f"prediction {out[debug_index_target]}")

        if num_targets == 0:
            print(f"num_targets is 0")
            compute_grad = False
            return compute_grad

        return loss / num_targets

    def apply(self, dataset: Dataset, patient_id: str):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + "_df_tensor").to(self.device)
                )

            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[
                patient_mask_index
            ]
            predictions = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1),
                device=self.device,
            )
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1),
                device=self.device,
            )
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1),
                dtype=torch.int64,
                device=self.device,
            )
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1),
                device=self.device,
            )

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v

                combined = {
                    event: torch.zeros(
                        size=(
                            1,
                            seq_lengths[visit][index].item(),
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
                    if seq_lengths[visit][index] > 0:

                        combined[event][
                            : seq_lengths[visit][index], :
                        ] = encoder_outputs[event][: seq_lengths[visit][index]]

                    else:
                        continue

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[
                    seq_lengths[visit, 0], dataset.target_value_index
                ]
                # TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[
                    seq_lengths[visit, 0], dataset.time_index
                ]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[
                    seq_lengths[visit, 0], dataset.target_index
                ]
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index].item() > 0:
                        combined_lstm[
                            0,
                            self.combined_history_size[1][
                                index
                            ] : self.combined_history_size[1][index + 1],
                        ] = self.lstm_modules[event](combined[event])[1][0][-1]
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device)
                )
                pred_input = torch.cat(
                    (
                        general_info,
                        combined_lstm[available_visit_mask[visit]].reshape(
                            1, combined_lstm[available_visit_mask[visit]].shape[2]
                        ),
                        time_to_targets[index_target],
                    ),
                    dim=1,
                )
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == "classification":
                predictions = torch.tensor(
                    [torch.argmax(elem) for elem in predictions], device=self.device
                )

        return predictions, "nothing", target_values, time_to_targets, target_categories
