import torch
import numpy as np
from scqm.custom_library.data_objects.masks import Masks

from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import EventEncoder
from scqm.custom_library.models.modules.transformer import TransformerSCQM
from scqm.custom_library.models.modules.lstms import (
    LstmEventSpecificPack,
)
from scqm.custom_library.models.modules.predictions import PredModule


from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch
from scqm.custom_library.utils import SaveOutput


class TransformerModel(Model):
    # multitask with sum instead of concat
    def __init__(self, config: dict, device: str):
        super().__init__(config, device)
        self.task = "regression"
        self.size_embedding = config["size_embedding"]
        # tuple storing the sum of the history sizes
        # and the cumulative sum of the history sizes, used as indices later to store the
        # outputs of the individual lstms in a combined vector.
        self.combined_history_size = (
            config["dim_val"] * len(config["event_names"]),
            np.cumsum([0] + [config["dim_val"] for name in config["event_names"]]),
        )
        self.config = config

        self.encoders = {
            name: EventEncoder(
                config[name]["num_features"],
                config["dim_val"],
                config["num_layers_enc"],
                config["hidden_enc"],
                config["dropout"],
            ).to(device)
            for name in config["event_names"]
        }

        self.p_encoder = EventEncoder(
            config["patients"]["num_features"],
            config["dim_val"],
            config["num_layers_enc"],
            config["hidden_enc"],
            config["dropout"],
        ).to(device)

        self.transformers = {
            name: TransformerSCQM(
                config["dim_val"],
                device,
                batch_first=config["batch_first"],
                n_heads=config["n_heads"],
                dropout=config["dropout"],
            ).to(device)
            for name in config["event_names"]
        }

        self.GlobalAttention = torch.nn.Parameter(
            torch.ones(size=(config["dim_val"], 1), device=self.device),
            requires_grad=True,
        )
        # + 1 for time to prediction
        self.pred_input_size = config["dim_val"] + 1
        self.PModuleDas28 = PredModule(
            self.pred_input_size,
            1,
            config["num_layers_pred"],
            config["hidden_pred"],
            config["dropout"],
        ).to(device)
        self.PModuleAsdas = PredModule(
            self.pred_input_size,
            1,
            config["num_layers_pred"],
            config["hidden_pred"],
            config["dropout"],
        ).to(device)
        self.batch_first = config["batch_first"]

        self.parameters = (
            list(self.PModuleDas28.parameters())
            + list(self.PModuleAsdas.parameters())
            + list(self.p_encoder.parameters())
            + [self.GlobalAttention]
        )
        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())
            self.parameters += list(self.transformers[name].parameters())

    def train(self):
        for name in self.encoders:
            self.encoders[name].train()
            self.transformers[name].train()
        self.p_encoder.train()
        self.PModuleDas28.train()
        self.PModuleAsdas.train()

    def eval(self):

        for name in self.encoders:
            self.encoders[name].eval()
            self.transformers[name].eval()
        self.p_encoder.eval()
        self.PModuleDas28.eval()
        self.PModuleAsdas.eval()

    def get_mask(self, lengths, device):
        max_length = max(lengths)
        mask = torch.zeros(
            size=(len(lengths), max_length), dtype=torch.bool, device=device
        )
        for index, elem in enumerate(lengths):
            mask[index, elem:] = 1
        return mask

    def apply_and_get_loss(
        self, dataset: Dataset, criterion: torch.nn, batch: Batch, target_name: str
    ):
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
        for v in range(0, batch.max_num_targets - dataset.min_num_targets + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_target_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            # sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros(
                (len(dataset.event_names),), dtype=torch.int32, device=self.device
            )
            # indices_lstm = torch.zeros(
            #     (len(dataset.event_names),), dtype=torch.int32, device=self.device
            # )
            if target_name == "das283bsr_score":
                target_index_in_events = dataset.event_names.index("a_visit")
                target_index_in_tensor = dataset.target_value_index_das28
                target_tensor = dataset.targets_das28_df_scaled_tensor_train
                batch_indices_targets = batch.indices_targets_das28
                time_index = dataset.time_index_das28
            else:
                target_index_in_events = dataset.event_names.index("a_visit")
                target_index_in_tensor = dataset.target_value_index_asdas
                target_tensor = dataset.targets_asdas_df_scaled_tensor_train
                batch_indices_targets = batch.indices_targets_asdas
                time_index = dataset.time_index_asdas
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_target_mask[:, v] == True).item(), 1),
                device=self.device,
            )
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_target_mask[:, v] == True).item(),),
                dtype=torch.int64,
                device=self.device,
            )
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_target_mask[:, v] == True).item(), 1),
                device=self.device,
            )

            index_target = 0
            debug_index_target = None
            # to contain encoder outputs. For each event, each patient is represented by a vector of shape max_num_events * embedding size
            # if a patient has less events, the last rows are zero padded
            combined = {
                event: torch.zeros(
                    size=(
                        len(batch.current_indices),
                        batch.seq_lengths[v][:, index].max().item(),
                        self.config["dim_val"],
                    ),
                    device=self.device,
                )
                for index, event in enumerate(dataset.event_names)
            }

            # combined representation of all lstm outputs (before prediction and after having applied lstms)
            # size: num_patients x size history
            combined_lstm = torch.zeros(
                size=(len(batch.current_indices), self.combined_history_size[0]),
                device=self.device,
            )

            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_target_mask[patient, v] == True:
                    # compute for each event the encoder outputs
                    for index, event in enumerate(dataset.event_names):
                        if seq[index] > 0:
                            combined[event][patient, : seq[index], :] = encoder_outputs[
                                event
                            ][indices[index] : indices[index] + seq[index]]
                        else:
                            continue
                    target_values[index_target] = target_tensor[batch_indices_targets][
                        indices[target_index_in_events]
                        + v
                        + dataset.min_num_targets
                        - 1,
                        target_index_in_tensor,
                    ]

                    time_to_targets[index_target] = target_tensor[
                        batch_indices_targets
                    ][
                        indices[target_index_in_events]
                        + v
                        + dataset.min_num_targets
                        - 1,
                        time_index,
                    ]

                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f"next target value: {target_values[index_target]}")
                        print(
                            f"Next target category : {target_categories[index_target]}"
                        )
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]
                    # indices_lstm[index] += 1
            # compute combined representation, for each event we fill the combined representation
            # in the dedicated columns

            for index, event in enumerate(dataset.event_names):
                # patients with visit to predict and available event
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                # patients = [
                #     elem.item()
                #     for elem in patients
                #     if batch.available_target_mask[elem, v]
                # ]

                if len(patients) > 0:
                    # compute the lengths of the sequences for each patient with available visit v
                    lengths = batch.seq_lengths[v, patients, index].cpu()

                    pad_sequence = torch.nn.utils.rnn.pad_sequence(
                        combined[event][patients, :, :],
                        batch_first=self.batch_first,
                    )
                    mask = self.get_mask(lengths, device=self.device)
                    # output = self.transformers[event](combined[event][patients, :, :])
                    output = self.transformers[event](pad_sequence, mask)

                    output = output[:, -1, :]
                    # output = torch.sum(output, dim=1)
                    combined_lstm[
                        patients,
                        self.combined_history_size[1][
                            index
                        ] : self.combined_history_size[1][index + 1],
                    ] = output
            # torch.reshape(combined_lstm[batch.available_target_mask[:, v]], shape=(len(torch.nonzero(batch.available_target_mask[:, v])), len(dataset.event_names), self.history_size))
            combined_lstm_input = torch.reshape(
                combined_lstm[batch.available_target_mask[:, v]],
                shape=(
                    len(torch.nonzero(batch.available_target_mask[:, v])),
                    len(dataset.event_names),
                    self.config["dim_val"],
                ),
            )
            # TODO here concat already with general patient info (has to have same size as self.history_size, potential problem ? Since history_size is larger than number of patient features)
            general_info = patient_encoding[batch.available_target_mask[:, v]]
            general_info = torch.reshape(
                general_info,
                shape=(
                    len(torch.nonzero(batch.available_target_mask[:, v])),
                    1,
                    self.config["dim_val"],
                ),
            )
            combined_input = torch.cat((general_info, combined_lstm_input), dim=1)
            global_attention_weights = torch.nn.Softmax(dim=1)(
                torch.matmul(combined_input, self.GlobalAttention)
            )
            combined_input = torch.sum(global_attention_weights * combined_input, dim=1)
            # general_info = patient_encoding[batch.available_target_mask[:, v]]
            pred_input = torch.cat(
                (
                    combined_input,
                    time_to_targets,
                ),
                dim=1,
            )

            # apply prediction module
            if target_name == "das283bsr_score":
                out = self.PModuleDas28(pred_input)
            else:
                out = self.PModuleAsdas(pred_input)

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

    def apply(
        self,
        dataset: Dataset,
        patient_id: str,
        target_name: str,
        return_history: bool = False,
    ):

        with torch.no_grad():
            self.eval()
            # method to directly apply the model to a single patient
            if target_name not in dataset[patient_id].targets_df.keys():
                raise ValueError("target name not available for this patient")
            else:
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

                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + "_df_tensor").to(self.device)
                )

            seq_lengths = masks.seq_lengths[:, patient_mask_index, :]
            available_target_mask = masks.available_target_mask[patient_mask_index]
            predictions = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1),
                device=self.device,
            )
            histories = torch.zeros(
                size=(
                    torch.sum(available_target_mask == True).item(),
                    self.pred_input_size,
                ),
                device=self.device,
            )
            histories_per_event = {
                event: torch.zeros(
                    size=(
                        torch.sum(available_target_mask == True).item(),
                        self.config["dim_val"],
                    ),
                    device=self.device,
                )
                for event in dataset.event_names
            }
            max_num_targets = masks.num_targets[patient_mask_index]
            total_num = masks.total_num[patient_mask_index]

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1),
                device=self.device,
            )
            time_to_targets = torch.empty(
                size=(torch.sum(available_target_mask == True).item(), 1, 1),
                device=self.device,
            )
            all_attention = {
                index: {event: [] for event in dataset.event_names}
                for index in range(torch.sum(available_target_mask == True).item())
            }
            all_global_attention = []

            index_target = 0
            prediction_dates = []
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
                            self.config["dim_val"],
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

                prediction_dates.append(
                    targets_df["date"].iloc[t + dataset.min_num_targets - 1]
                )
                # target_categories[index_target] = dataset[patient_id].targets_df_tensor[
                #     visit + dataset.min_num_targets - 1, dataset.target_index
                # ]
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[t][index].item() > 0:
                        lengths = seq_lengths[t][index].reshape(1)
                        mask = self.get_mask(lengths, device=self.device)

                        pad_sequence = torch.nn.utils.rnn.pad_sequence(
                            combined[event], batch_first=self.batch_first
                        )

                        output = self.transformers[event](pad_sequence, mask)
                        # output = torch.sum(output, dim=1)

                        output = output[:, -1, :]
                        combined_lstm[
                            0,
                            self.combined_history_size[1][
                                index
                            ] : self.combined_history_size[1][index + 1],
                        ] = output

                combined_lstm_input = torch.reshape(
                    combined_lstm[0],
                    shape=(
                        1,
                        len(dataset.event_names),
                        self.config["dim_val"],
                    ),
                )
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device)
                ).reshape(1, 1, self.config["dim_val"])
                combined_input = torch.cat((general_info, combined_lstm_input), dim=1)
                global_attention_weights = torch.nn.Softmax(dim=1)(
                    torch.matmul(combined_input, self.GlobalAttention)
                )
                all_global_attention.append(global_attention_weights)
                combined_input = torch.sum(
                    global_attention_weights * combined_input, dim=1
                )

                pred_input = torch.cat(
                    (
                        combined_input,
                        time_to_targets[index_target],
                    ),
                    dim=1,
                )
                histories[index_target] = pred_input
                if target_name == "das283bsr_score":
                    predictions[index_target] = self.PModuleDas28(pred_input)
                else:
                    predictions[index_target] = self.PModuleAsdas(pred_input)
                index_target += 1
        if return_history:
            return (
                predictions,
                target_values,
                time_to_targets,
                histories,
                histories_per_event,
                all_attention,
                all_global_attention,
            )

        else:
            return (predictions, target_values, time_to_targets, prediction_dates)
