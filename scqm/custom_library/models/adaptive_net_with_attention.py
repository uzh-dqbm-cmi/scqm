import torch

from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import PaddedEventEncoder
from scqm.custom_library.models.modules.lstms import LstmAllHistory
from scqm.custom_library.models.modules.predictions import PredModule

class AdaptivenetWithAttention(Model):
    def __init__(self, config, device, modules=None):
        super().__init__(config, device)
        if modules is None:
            self.pretraining = False
        else:
            self.pretraining = True
        self.size_embedding = config["size_embedding"]
        self.num_targets = config["num_targets"]
        # self.encoders = {name: Encoder(model_specifics[name]['num_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
        #                                model_specifics['hidden_enc'], model_specifics['dropout']).to(device) for name in model_specifics['event_names']}
        if self.pretraining:
            self.encoders = modules["encoders"]
            self.p_encoder = modules["p_encoder"]
            self.LModule = modules["LModule"]

        else:
            self.encoders = {
                name: PaddedEventEncoder(
                    config[name]["num_features"],
                    config[name]["size_out"],
                    config["size_embedding"],
                    config["num_layers_enc"],
                    config["hidden_enc"],
                    config["dropout"],
                ).to(device)
                for name in config["event_names"]
            }

            self.p_encoder = PaddedEventEncoder(
                config["patients"]["num_features"],
                config["patients"]["size_out"],
                config["size_embedding"],
                config["num_layers_enc"],
                config["hidden_enc"],
                config["dropout"],
            ).to(device)

            self.LModule = LstmAllHistory(
                config["size_embedding"],
                config["device"],
                config["batch_first"],
                config["size_history"],
                config["num_layers"],
            ).to(device)
            self.Attention = torch.nn.Parameter(
                torch.ones(size=(config["size_history"], 1), device=self.device),
                requires_grad=True,
            )
        # + 1 for time to prediction
        # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
        #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.PModule = PredModule(
            config["size_history"] + config["size_embedding"] + 1,
            config["num_targets"],
            config["num_layers_pred"],
            config["hidden_pred"],
            config["dropout"],
        ).to(device)
        self.task = config["task"]
        self.batch_first = config["batch_first"]

        if self.pretraining:
            self.parameters = list(self.PModule.parameters()) + [self.Attention]
        else:
            self.parameters = (
                list(self.LModule.parameters())
                + list(self.PModule.parameters())
                + list(self.p_encoder.parameters())
                + [self.Attention]
            )
            for name in self.encoders:
                self.parameters += list(self.encoders[name].parameters())

    def train(self):
        for name in self.encoders:
            self.encoders[name].train()
        self.p_encoder.train()
        self.LModule.train()
        self.PModule.train()

    def eval(self):

        for name in self.encoders:
            self.encoders[name].eval()
        self.p_encoder.eval()
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, criterion, batch):
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
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros(
                (len(dataset.event_names),), dtype=torch.int32, device=self.device
            )
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
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v
                    combined = torch.zeros(
                        size=(seq.sum(), self.size_embedding), device=self.device
                    )
                    for index, event in enumerate(dataset.event_names):
                        mask = getattr(batch, event + "_masks")
                        combined[
                            torch.broadcast_to(
                                mask[patient][v],
                                (len(mask[patient][v]), self.size_embedding),
                            )
                        ] = encoder_outputs[event][
                            indices[index]: indices[index] + seq[index]
                        ].flatten()
                    sequence.append(combined)
                    target_values[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + seq[visit_index],
                        dataset.target_value_index,
                    ]
                    # TODO change
                    time_to_targets[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + seq[visit_index], dataset.time_index
                    ]
                    target_categories[
                        index_target
                    ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
                        indices[visit_index] + seq[visit_index], dataset.target_index
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
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(
                sequence, batch_first=self.batch_first
            )
            # compute the lengths of the sequences for each patient with available visit v
            lengths = (
                batch.seq_lengths[v].sum(dim=1)[batch.available_visit_mask[:, v]].cpu()
            )

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence,
                batch_first=self.batch_first,
                lengths=lengths,
                enforce_sorted=False,
            )

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            # history = hn[-1]
            # compute unpacked output
            unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=self.batch_first
            )[0]
            # compute attention weights
            attention_weights = torch.nn.Softmax(dim=1)(
                torch.matmul(unpacked_output, self.Attention)
            )
            # weighted history
            history = torch.sum(unpacked_output * attention_weights, dim=1)
            # concat computed patient history with general information
            # general_info = self.p_encoder(
            #     dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]])
            general_info = patient_encoding[batch.available_visit_mask[:, v]]

            pred_input = torch.cat((general_info, history, time_to_targets), dim=1)
            # pred_input = torch.cat(
            #     (dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]], history, time_to_targets), dim=1)
            # apply prediction module
            out = self.PModule(pred_input)
            # compute loss
            if self.task == "classification":
                targets = target_categories
            else:
                targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)
            # # compute other training metrics
            # if self.task == 'classification':
            #     predictions = torch.tensor([torch.argmax(elem) for elem in out], device=self.device)
            # else:
            #     predictions = out
            # metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f"prediction {out[debug_index_target]}")

        return loss / num_targets

    def apply(self, dataset, patient_id):
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
                size=(torch.sum(available_visit_mask == True).item(), self.num_targets),
                device=self.device,
            )
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]
            all_history = torch.empty(
                size=(
                    max_num_visits - dataset.min_num_visits + 1,
                    self.LModule.hidden_size,
                ),
                device=self.device,
            )

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
                combined = torch.zeros(
                    size=(seq_lengths[visit].sum(), self.size_embedding),
                    device=self.device,
                )
                for index, event in enumerate(dataset.event_names):
                    mask = getattr(dataset.masks, event + "_masks")[patient_mask_index]
                    combined[
                        torch.broadcast_to(
                            mask[visit], (len(mask[visit]), self.size_embedding)
                        )
                    ] = encoder_outputs[event][: seq_lengths[visit][index]].flatten()

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
                # apply lstm
                padded_sequence = torch.nn.utils.rnn.pad_sequence(
                    [combined], batch_first=self.batch_first
                )
                lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
                pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence,
                    batch_first=self.batch_first,
                    lengths=lengths,
                    enforce_sorted=False,
                )
                output, (hn, cn) = self.LModule(pack_padded)
                # unpacked output
                unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=self.batch_first
                )[0]
                attention_weights = torch.nn.Softmax(dim=1)(
                    torch.matmul(unpacked_output, self.Attention)
                )
                # weighted history
                history = torch.sum(unpacked_output * attention_weights, dim=1)
                # history = hn[-1]
                all_history[visit, :] = history
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device)
                )
                pred_input = torch.cat(
                    (general_info, history, time_to_targets[index_target]), dim=1
                )
                # pred_input = torch.cat(
                #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
                # apply prediction module
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == "classification":
                predictions = torch.tensor(
                    [torch.argmax(elem) for elem in predictions], device=self.device
                )

        return (
            predictions,
            all_history,
            target_values,
            time_to_targets,
            target_categories,
        )
