import torch

from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import PaddedEncoderFixed
from scqm.custom_library.models.modules.decoders import PaddedDecoderFixed
from scqm.custom_library.models.modules.lstms import LstmAllHistory
from scqm.custom_library.models.modules.predictions import PredModule
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch


class AutoEncoder(Model):
    """ " Autoencoder for feature representation learning"""

    def __init__(self, config: dict, device: str):
        """instantiate model

        Args:
            config (dict): model parameters
            device (str): device
        """
        super().__init__(device)
        self.size_embedding = config["size_embedding"]
        self.encoders = {
            name: PaddedEncoderFixed(
                config[name]["num_features"],
                config[name]["size_out"],
                config["size_embedding"],
                config["num_layers_enc"],
                config["dropout"],
            ).to(device)
            for name in config["event_names"]
        }
        self.decoders = {
            name: PaddedDecoderFixed(
                config["size_embedding"],
                config[name]["num_features"],
                config["num_layers_enc"],
            ).to(device)
            for name in config["event_names"]
        }
        self.p_encoder = PaddedDecoderFixed(
            config["patients"]["num_features"],
            config["patients"]["size_out"],
            config["size_embedding"],
            config["num_layers_enc"],
            config["dropout"],
        ).to(device)
        self.p_decoder = PaddedDecoderFixed(
            config["size_embedding"],
            config["patients"]["num_features"],
            config["num_layers_enc"],
        ).to(device)

        self.parameters = list(self.p_encoder.parameters()) + list(
            self.p_decoder.parameters()
        )

        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())
            self.parameters += list(self.decoders[name].parameters())

    def train(self):
        """put model in train mode"""
        for name in self.encoders:
            self.encoders[name].train()
            self.decoders[name].train()
        self.p_encoder.train()
        self.p_decoder.train()

    def eval(self):
        """put model in evaluation mode"""
        for name in self.encoders:
            self.encoders[name].eval()
            self.decoders[name].eval()
        self.p_encoder.eval()
        self.p_decoder.eval()

    def apply_and_get_loss(self, dataset: Dataset, criterion: torch.nn, batch: Batch):
        """apply model to batch of data and get loss

        Args:
            dataset (Dataset): dataset
            criterion (torch.nn): loss criterion
            batch (Batch): batch

        Returns:
            _type_: loss
        """
        encoder_outputs = {}
        decoder_outputs = {}
        all_losses = {}
        loss = 0
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, "indices_" + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + "_df_scaled_tensor_train")[indices]
                )
                decoder_outputs[event] = self.decoders[event](encoder_outputs[event])

                loss += criterion(
                    getattr(dataset, event + "_df_scaled_tensor_train")[indices],
                    decoder_outputs[event],
                )
                all_losses[event] = criterion(
                    getattr(dataset, event + "_df_scaled_tensor_train")[indices],
                    decoder_outputs[event],
                )
        p_encoder_output = self.p_encoder(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients]
        )
        p_decoder_output = self.p_decoder(p_encoder_output)
        loss += criterion(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients],
            p_decoder_output,
        )
        # if loss != loss:
        #     for event in dataset.event_names:
        #         print(f' {event} {len(getattr(batch, "indices_" + event))}')

        all_losses["patients"] = criterion(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients],
            p_decoder_output,
        )

        return loss, all_losses


# class AutoEncoderAdaNet(Model):
#     def __init__(self, model_specifics, device):
#         super().__init__(device)
#         self.size_embedding = model_specifics["size_embedding"]
#         self.num_targets = model_specifics["num_targets"]
#         self.encoders = {
#             name: PaddedEncoderBis(
#                 model_specifics[name]["num_features"],
#                 model_specifics[name]["size_out"],
#                 model_specifics["size_embedding"],
#                 model_specifics["num_layers_enc"],
#                 model_specifics["dropout"],
#             ).to(device)
#             for name in model_specifics["event_names"]
#         }
#         self.decoders = {
#             name: DecoderBis(
#                 model_specifics["size_embedding"],
#                 model_specifics[name]["num_features"],
#                 model_specifics["num_layers_enc"],
#             ).to(device)
#             for name in model_specifics["event_names"]
#         }
#         self.p_encoder = PaddedEncoderBis(
#             model_specifics["patients"]["num_features"],
#             model_specifics["patients"]["size_out"],
#             model_specifics["size_embedding"],
#             model_specifics["num_layers_enc"],
#             model_specifics["dropout"],
#         ).to(device)
#         self.p_decoder = DecoderBis(
#             model_specifics["size_embedding"],
#             model_specifics["patients"]["num_features"],
#             model_specifics["num_layers_enc"],
#         ).to(device)
#         self.LModule = LSTMModule(
#             model_specifics["size_embedding"],
#             model_specifics["device"],
#             model_specifics["batch_first"],
#             model_specifics["size_history"],
#             model_specifics["num_layers"],
#         ).to(device)
#         # + 1 for time to prediction
#         # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
#         #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
#         self.PModule = PredModule(
#             model_specifics["size_history"] + model_specifics["size_embedding"] + 1,
#             model_specifics["num_targets"],
#             model_specifics["num_layers_pred"],
#             model_specifics["hidden_pred"],
#             model_specifics["dropout"],
#         ).to(device)
#         self.task = model_specifics["task"]
#         self.batch_first = model_specifics["batch_first"]

#         self.parameters = (
#             list(self.p_encoder.parameters())
#             + list(self.p_decoder.parameters())
#             + list(self.LModule.parameters())
#             + list(self.PModule.parameters())
#         )

#         for name in self.encoders:
#             self.parameters += list(self.encoders[name].parameters())
#             self.parameters += list(self.decoders[name].parameters())

#     def train(self):

#         for name in self.encoders:
#             self.encoders[name].train()
#             self.decoders[name].train()
#         self.p_encoder.train()
#         self.p_decoder.train()
#         self.LModule.train()
#         self.PModule.train()

#     def eval(self):
#         for name in self.encoders:
#             self.encoders[name].eval()
#             self.decoders[name].eval()
#         self.p_encoder.eval()
#         self.p_decoder.eval()
#         self.LModule.eval()
#         self.PModule.eval()

#     def apply_and_get_loss(self, dataset, decoding_criterion, pred_criterion, batch):
#         encoder_outputs = {}
#         decoder_outputs = {}
#         all_losses = {}
#         decoding_loss = 0
#         # apply encoders

#         for event in dataset.event_names:
#             indices = getattr(batch, "indices_" + event)
#             if len(indices) > 0:
#                 encoder_outputs[event] = self.encoders[event](
#                     getattr(dataset, event + "_df_scaled_tensor_train")[indices]
#                 )
#                 decoder_outputs[event] = self.decoders[event](encoder_outputs[event])

#                 decoding_loss += decoding_criterion(
#                     getattr(dataset, event + "_df_scaled_tensor_train")[indices],
#                     decoder_outputs[event],
#                 )
#                 all_losses[event] = decoding_criterion(
#                     getattr(dataset, event + "_df_scaled_tensor_train")[indices],
#                     decoder_outputs[event],
#                 )
#             else:
#                 encoder_outputs[event] = torch.tensor([], device=self.device)

#         p_encoder_output = self.p_encoder(
#             dataset.patients_df_scaled_tensor_train[batch.indices_patients]
#         )
#         p_decoder_output = self.p_decoder(p_encoder_output)
#         decoding_loss += decoding_criterion(
#             dataset.patients_df_scaled_tensor_train[batch.indices_patients],
#             p_decoder_output,
#         )

#         # for scaling of loss
#         prediction_loss = 0
#         num_targets = 0
#         for v in range(0, batch.max_num_targets - dataset.min_num_targets + 1):
#             # stores for all the patients in the batch the tensor of ordered events (of varying size)
#             sequence = []
#             # to keep track of the right index in the visits/medication tensors
#             indices = torch.zeros(
#                 (len(dataset.event_names),), dtype=torch.int32, device=self.device
#             )
#             visit_index = dataset.event_names.index("a_visit")
#             # targets (values)
#             target_values = torch.empty(
#                 size=(torch.sum(batch.available_target_mask[:, v] == True).item(), 1),
#                 device=self.device,
#             )
#             # targets caetgories
#             target_categories = torch.empty(
#                 size=(torch.sum(batch.available_target_mask[:, v] == True).item(),),
#                 dtype=torch.int64,
#                 device=self.device,
#             )
#             # delta t
#             time_to_targets = torch.empty(
#                 size=(torch.sum(batch.available_target_mask[:, v] == True).item(), 1),
#                 device=self.device,
#             )
#             # for each patient combine the medication and visit events in the right order up to visit v
#             index_target = 0
#             debug_index_target = None
#             for patient, seq in enumerate(batch.seq_lengths[v]):
#                 # check if the patient has at least v visits
#                 if batch.available_target_mask[patient, v] == True:
#                     # create combined ordered list of visit/medication/events up to v
#                     combined = torch.zeros(
#                         size=(seq.sum(), self.size_embedding), device=self.device
#                     )
#                     for index, event in enumerate(dataset.event_names):
#                         mask = getattr(batch, event + "_masks")
#                         combined[
#                             torch.broadcast_to(
#                                 mask[patient][v],
#                                 (len(mask[patient][v]), self.size_embedding),
#                             )
#                         ] = encoder_outputs[event][
#                             indices[index]: indices[index] + seq[index]
#                         ].flatten()
#                     sequence.append(combined)
#                     target_values[
#                         index_target
#                     ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
#                         indices[visit_index] + seq[visit_index],
#                         dataset.target_value_index,
#                     ]
#                     time_to_targets[
#                         index_target
#                     ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
#                         indices[visit_index] + seq[visit_index], dataset.time_index
#                     ]
#                     target_categories[
#                         index_target
#                     ] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][
#                         indices[visit_index] + seq[visit_index], dataset.target_index
#                     ]
#                     if batch.debug_index != None and patient == batch.debug_index:
#                         print(f"next target value: {target_values[index_target]}")
#                         print(
#                             f"Next target category : {target_categories[index_target]}"
#                         )
#                         debug_index_target = index_target
#                     index_target += 1
#                 # update the indices to select from in the tensors
#                 for index, event in enumerate(dataset.event_names):
#                     indices[index] += batch.total_num[patient, index]
#             # "preprocessing" to apply lstm
#             padded_sequence = torch.nn.utils.rnn.pad_sequence(
#                 sequence, batch_first=self.batch_first
#             )
#             # compute the lengths of the sequences for each patient with available visit v
#             lengths = (
#                 batch.seq_lengths[v].sum(dim=1)[batch.available_target_mask[:, v]].cpu()
#             )

#             pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
#                 padded_sequence,
#                 batch_first=self.batch_first,
#                 lengths=lengths,
#                 enforce_sorted=False,
#             )

#             # apply lstm
#             output, (hn, cn) = self.LModule(pack_padded_sequence)
#             history = hn[-1]
#             # concat computed patient history with general information
#             # general_info = self.p_encoder(
#             #     dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_target_mask[:, v]])
#             general_info = p_encoder_output[batch.available_target_mask[:, v]]

#             pred_input = torch.cat((general_info, history, time_to_targets), dim=1)
#             # pred_input = torch.cat(
#             #     (dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_target_mask[:, v]], history, time_to_targets), dim=1)
#             # apply prediction module
#             out = self.PModule(pred_input)
#             # compute loss
#             if self.task == "classification":
#                 targets = target_categories
#             else:
#                 targets = target_values
#             prediction_loss += pred_criterion(out, targets)
#             num_targets += len(targets)
#             # # compute other training metrics
#             # if self.task == 'classification':
#             #     predictions = torch.tensor([torch.argmax(elem) for elem in out], device=self.device)
#             # else:
#             #     predictions = out
#             # metrics.add_observations(predictions, targets)

#             if debug_index_target != None:
#                 # print(out)
#                 print(f"prediction {out[debug_index_target]}")

#         prediction_loss = prediction_loss / num_targets

#         return decoding_loss, prediction_loss

#     def apply(self, dataset, patient_id):

#         with torch.no_grad():
#             # method to directly apply the model to a single patient
#             patient_mask_index = dataset.mapping_for_masks[patient_id]
#             encoder_outputs = {}
#             for event in dataset.event_names:
#                 encoder_outputs[event] = self.encoders[event](
#                     getattr(dataset[patient_id], event + "_df_tensor").to(self.device)
#                 )

#             predictions = torch.empty(
#                 size=(
#                     len(dataset[patient_id].visits) - dataset.min_num_targets + 1,
#                     self.num_targets,
#                 ),
#                 device=self.device,
#             )
#             seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]

#             available_target_mask = dataset.masks.available_target_mask[
#                 patient_mask_index
#             ]
#             max_num_targets = dataset.masks.num_visits[patient_mask_index]
#             total_num = dataset.masks.total_num[patient_mask_index]
#             all_history = torch.empty(
#                 size=(
#                     max_num_targets - dataset.min_num_targets + 1,
#                     self.LModule.hidden_size,
#                 ),
#                 device=self.device,
#             )
#             for visit in range(0, max_num_targets - dataset.min_num_targets + 1):
#                 # create combined ordered list of visit/medication/events up to v
#                 combined = torch.zeros(
#                     size=(seq_lengths[visit].sum(), self.size_embedding),
#                     device=self.device,
#                 )
#                 for index, event in enumerate(dataset.event_names):
#                     mask = getattr(dataset.masks, event + "_masks")[patient_mask_index]
#                     combined[
#                         torch.broadcast_to(
#                             mask[visit], (len(mask[visit]), self.size_embedding)
#                         )
#                     ] = encoder_outputs[event][: seq_lengths[visit][index]].flatten()
#                 time_to_target = torch.tensor(
#                     [[dataset[patient_id].targets_df_tensor[visit - 1, 0]]],
#                     device=self.device,
#                 )
#                 # apply lstm
#                 padded_sequence = torch.nn.utils.rnn.pad_sequence(
#                     [combined], batch_first=self.batch_first
#                 )
#                 lengths = seq_lengths[visit].sum()[available_target_mask[visit]].cpu()
#                 pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
#                     padded_sequence,
#                     batch_first=self.batch_first,
#                     lengths=lengths,
#                     enforce_sorted=False,
#                 )
#                 output, (hn, cn) = self.LModule(pack_padded)
#                 history = hn[-1]
#                 all_history[visit, :] = history
#                 # concat computed patient history with general information
#                 general_info = self.p_encoder(
#                     dataset[patient_id].patients_df_tensor.to(self.device)
#                 )
#                 pred_input = torch.cat((general_info, history, time_to_target), dim=1)
#                 # pred_input = torch.cat(
#                 #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
#                 # apply prediction module
#                 predictions[visit - dataset.min_num_targets] = self.PModule(pred_input)
#             if self.task == "classification":
#                 predictions = torch.tensor(
#                     [torch.argmax(elem) for elem in predictions], device=self.device
#                 )

#         return predictions, all_history
