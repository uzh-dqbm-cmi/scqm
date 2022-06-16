import torch
import numpy as np

from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import EventEncoder

from scqm.custom_library.models.modules.lstms import LstmEventSpecific
from scqm.custom_library.models.modules.predictions import PredModule

from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch


class Othernet(Model):
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

    def apply_and_get_loss(
        self, dataset: Dataset, criterion: torch.nn, batch: Batch
    ) -> torch.tensor:
        """Apply model to batch of data and get loss

        Args:
            dataset (Dataset): dataset
            criterion (torch.nn): loss criterion for optimizer
            batch (Batch): batch

        Returns:
            torch.tensor: total loss divided by number of targets
        """
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
            # indices_lstm = torch.zeros(
            #     (len(dataset.event_names),), dtype=torch.int32, device=self.device
            # )
            visit_index = dataset.event_names.index("a_visit")
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1),
                device=self.device,
            )
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),),
                dtype=torch.int64,
                device=self.device,
            )
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1),
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
                        self.config[event]["size_out"],
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
                if batch.available_visit_mask[patient, v] == True:
                    # compute for each event the encoder outputs
                    for index, event in enumerate(dataset.event_names):
                        if seq[index] > 0:
                            combined[event][patient, : seq[index], :] = encoder_outputs[
                                event
                            ][indices[index] : indices[index] + seq[index]]
                        else:
                            continue
                    # indices[visit_index] + v + dataset.min_num_visits - 1
                    target_values[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + v + dataset.min_num_visits - 1,
                        dataset.target_value_index,
                    ]
                    # TODO change
                    time_to_targets[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + v + dataset.min_num_visits - 1,
                        dataset.time_index,
                    ]
                    target_categories[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + v + dataset.min_num_visits - 1,
                        dataset.target_index,
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
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                if len(patients) > 0:
                    combined_lstm[
                        batch.available_visit_mask[:, v],
                        self.combined_history_size[1][
                            index
                        ] : self.combined_history_size[1][index + 1],
                    ] = self.lstm_modules[event](
                        combined[event][batch.available_visit_mask[:, v]]
                    )[
                        1
                    ][
                        0
                    ][
                        -1
                    ]
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

        return loss / num_targets

    def apply(self, dataset: Dataset, patient_id: str):
        """apply model to a selected

        Args:
            dataset (Dataset): dataset
            patient_id (str): patient id

        Returns:
            _type_: _description_
        """
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
            visit_ids = []
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
            visit_ids = []
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
                    visit + dataset.min_num_visits - 1, dataset.target_value_index
                ]

                # TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[
                    visit + dataset.min_num_visits - 1, dataset.time_index
                ]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[
                    visit + dataset.min_num_visits - 1, dataset.target_index
                ]
                visit_ids.append(
                    dataset[patient_id]
                    .targets_df.iloc[visit + dataset.min_num_visits - 1]
                    .uid_num
                )
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

        return (
            predictions,
            "nothing",
            target_values,
            time_to_targets,
            target_categories,
            visit_ids,
        )
