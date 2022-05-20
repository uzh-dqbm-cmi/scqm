from abc import ABC, abstractmethod
import torch
from modules import Encoder, PaddedEncoder, LSTMModule, PredModule, Decoder, HistoryDecoder, PaddedEncoderBis, DecoderBis, LSTMModulebis
import numpy as np

# base model class


class Model(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    @abstractmethod
    def train(self):
        pass

    def eval(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError


class Adaptivenet(Model):
    def __init__(self, config, device, modules=None):
        super().__init__(config, device)
        if modules is None:
            self.pretraining = False
        else:
            self.pretraining = True
        self.size_embedding = config['size_embedding']
        self.num_targets = config['num_targets']
        # self.encoders = {name: Encoder(model_specifics[name]['num_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
        #                                model_specifics['hidden_enc'], model_specifics['dropout']).to(device) for name in model_specifics['event_names']}
        if self.pretraining:
            self.encoders = modules['encoders']
            self.p_encoder = modules['p_encoder']
            self.LModule = modules['LModule']

        else:
            self.encoders = {name: PaddedEncoder(config[name]['num_features'], config[name]['size_out'], config['size_embedding'], config['num_layers_enc'],
                                                 config['hidden_enc'], config['dropout']).to(device) for name in config['event_names']}

            self.p_encoder = PaddedEncoder(config['patients']['num_features'], config['patients']['size_out'], config['size_embedding'], config['num_layers_enc'],
                                           config['hidden_enc'], config['dropout']).to(device)

            self.LModule = LSTMModule(config['size_embedding'], config['device'],
                                      config['batch_first'], config['size_history'], config['num_layers']).to(device)
        # + 1 for time to prediction
        # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
        #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.PModule = PredModule(config['size_history'] + config['size_embedding'] +
                                  1, config['num_targets'], config['num_layers_pred'], config['hidden_pred'], config['dropout']).to(device)
        self.task = config['task']
        self.batch_first = config['batch_first']

        if self.pretraining:
            self.parameters = list(self.PModule.parameters())
        else:
            self.parameters = list(self.LModule.parameters()) + \
                list(self.PModule.parameters()) + list(self.p_encoder.parameters())
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
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype=torch.int64, device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v
                    combined = torch.zeros(size=(seq.sum(), self.size_embedding), device=self.device)
                    for index, event in enumerate(dataset.event_names):
                        mask = getattr(batch, event + '_masks')
                        combined[torch.broadcast_to(mask[patient][v], (len(mask[patient][v]), self.size_embedding))
                                 ] = encoder_outputs[event][indices[index]:indices[index] + seq[index]].flatten()
                    sequence.append(combined)
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                seq[visit_index], dataset.target_value_index]
                    #TODO change
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                  seq[visit_index], dataset.time_index]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                    seq[visit_index], dataset.target_index]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=self.batch_first)
            # compute the lengths of the sequences for each patient with available visit v
            lengths = batch.seq_lengths[v].sum(dim=1)[batch.available_visit_mask[:, v]].cpu()

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            history = hn[-1]
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
            if self.task == 'classification':
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
            #metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets

    def apply(self, dataset, patient_id):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            
            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            predictions = torch.empty(size=(torch.sum(available_visit_mask == True).item(), self.num_targets), device=self.device)
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]
            all_history = torch.empty(size=(max_num_visits - dataset.min_num_visits +
                                    1, self.LModule.hidden_size), device=self.device)
            
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(),1), dtype=torch.int64, device=self.device)
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1), device=self.device)

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v
                combined = torch.zeros(size=(seq_lengths[visit].sum(), self.size_embedding), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    mask = getattr(dataset.masks, event + '_masks')[patient_mask_index]
                    combined[torch.broadcast_to(mask[visit], (len(mask[visit]), self.size_embedding))
                            ] = encoder_outputs[event][:seq_lengths[visit][index]].flatten()

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0], dataset.target_value_index]
                #TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                    dataset.time_index]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                        dataset.target_index]
                # apply lstm
                padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
                lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
                pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
                output, (hn, cn) = self.LModule(pack_padded)
                history = hn[-1]
                all_history[visit, :] = history
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat((general_info, history, time_to_targets[index_target]), dim=1)
                # pred_input = torch.cat(
                #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
                # apply prediction module
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, all_history, target_values, time_to_targets, target_categories


class AdaptivenetWithAttention(Model):
    def __init__(self, config, device, modules=None):
        super().__init__(config, device)
        if modules is None:
            self.pretraining = False
        else:
            self.pretraining = True
        self.size_embedding = config['size_embedding']
        self.num_targets = config['num_targets']
        # self.encoders = {name: Encoder(model_specifics[name]['num_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
        #                                model_specifics['hidden_enc'], model_specifics['dropout']).to(device) for name in model_specifics['event_names']}
        if self.pretraining:
            self.encoders = modules['encoders']
            self.p_encoder = modules['p_encoder']
            self.LModule = modules['LModule']

        else:
            self.encoders = {name: PaddedEncoder(config[name]['num_features'], config[name]['size_out'], config['size_embedding'], config['num_layers_enc'],
                                                 config['hidden_enc'], config['dropout']).to(device) for name in config['event_names']}

            self.p_encoder = PaddedEncoder(config['patients']['num_features'], config['patients']['size_out'], config['size_embedding'], config['num_layers_enc'],
                                           config['hidden_enc'], config['dropout']).to(device)

            self.LModule = LSTMModule(config['size_embedding'], config['device'],
                                      config['batch_first'], config['size_history'], config['num_layers']).to(device)
            self.Attention = torch.nn.Parameter(torch.ones(size=(config['size_history'], 1), device=self.device), requires_grad=True)
        # + 1 for time to prediction
        # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
        #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.PModule = PredModule(config['size_history'] + config['size_embedding'] +
                                  1, config['num_targets'], config['num_layers_pred'], config['hidden_pred'], config['dropout']).to(device)
        self.task = config['task']
        self.batch_first = config['batch_first']

        if self.pretraining:
            self.parameters = list(self.PModule.parameters()) + [self.Attention]
        else:
            self.parameters = list(self.LModule.parameters()) + \
                list(self.PModule.parameters()) + list(self.p_encoder.parameters()) + [self.Attention]
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
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype=torch.int64, device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v
                    combined = torch.zeros(size=(seq.sum(), self.size_embedding), device=self.device)
                    for index, event in enumerate(dataset.event_names):
                        mask = getattr(batch, event + '_masks')
                        combined[torch.broadcast_to(mask[patient][v], (len(mask[patient][v]), self.size_embedding))
                                 ] = encoder_outputs[event][indices[index]:indices[index] + seq[index]].flatten()
                    sequence.append(combined)
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                seq[visit_index], dataset.target_value_index]
                    #TODO change
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                  seq[visit_index], dataset.time_index]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                    seq[visit_index], dataset.target_index]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=self.batch_first)
            # compute the lengths of the sequences for each patient with available visit v
            lengths = batch.seq_lengths[v].sum(dim=1)[batch.available_visit_mask[:, v]].cpu()

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            #history = hn[-1]
            # compute unpacked output
            unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)[0]
            # compute attention weights
            attention_weights = torch.nn.Softmax(dim=1)(torch.matmul(unpacked_output, self.Attention))
            # weighted history
            history= torch.sum(unpacked_output * attention_weights, dim=1)
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
            if self.task == 'classification':
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
            #metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets

    def apply(self, dataset, patient_id):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            predictions = torch.empty(size=(torch.sum(available_visit_mask == True).item(),
                                      self.num_targets), device=self.device)
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]
            all_history = torch.empty(size=(max_num_visits - dataset.min_num_visits +
                                            1, self.LModule.hidden_size), device=self.device)

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), dtype=torch.int64, device=self.device)
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1), device=self.device)

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v
                combined = torch.zeros(size=(seq_lengths[visit].sum(), self.size_embedding), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    mask = getattr(dataset.masks, event + '_masks')[patient_mask_index]
                    combined[torch.broadcast_to(mask[visit], (len(mask[visit]), self.size_embedding))
                             ] = encoder_outputs[event][:seq_lengths[visit][index]].flatten()

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                    dataset.target_value_index]
                #TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                      dataset.time_index]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                        dataset.target_index]
                # apply lstm
                padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
                lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
                pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
                output, (hn, cn) = self.LModule(pack_padded)
                # unpacked output
                unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)[0]
                attention_weights = torch.nn.Softmax(dim=1)(torch.matmul(unpacked_output, self.Attention))
                # weighted history
                history = torch.sum(unpacked_output * attention_weights, dim=1)
                #history = hn[-1]
                all_history[visit, :] = history
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat((general_info, history, time_to_targets[index_target]), dim=1)
                # pred_input = torch.cat(
                #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
                # apply prediction module
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, all_history, target_values, time_to_targets, target_categories


class AutoEncoder(Model):
    def __init__(self, model_specifics, device):
        super().__init__(device)
        self.size_embedding = model_specifics['size_embedding']
        self.encoders = {name: PaddedEncoderBis(model_specifics[name]['num_features'], model_specifics[name]['size_out'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                                model_specifics['dropout']).to(device) for name in model_specifics['event_names']}
        self.decoders = {name: DecoderBis(model_specifics['size_embedding'], model_specifics[name]['num_features'], model_specifics['num_layers_enc'],
                                          ).to(device) for name in model_specifics['event_names']}
        self.p_encoder = PaddedEncoderBis(model_specifics['patients']['num_features'], model_specifics['patients']['size_out'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                          model_specifics['dropout']).to(device)
        self.p_decoder = DecoderBis(model_specifics['size_embedding'], model_specifics['patients']
                                    ['num_features'], model_specifics['num_layers_enc']).to(device)

        self.parameters = list(self.p_encoder.parameters()) + list(self.p_decoder.parameters())

        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())
            self.parameters += list(self.decoders[name].parameters())

    def train(self):

        for name in self.encoders:
            self.encoders[name].train()
            self.decoders[name].train()
        self.p_encoder.train()
        self.p_decoder.train()

    def eval(self):
        for name in self.encoders:
            self.encoders[name].eval()
            self.decoders[name].eval()
        self.p_encoder.eval()
        self.p_decoder.eval()

    def apply_and_get_loss(self, dataset, criterion, batch):
        encoder_outputs = {}
        decoder_outputs = {}
        all_losses = {}
        loss = 0
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
                decoder_outputs[event] = self.decoders[event](encoder_outputs[event])

                loss += criterion(getattr(dataset, event + '_df_scaled_tensor_train')[indices], decoder_outputs[event])
                all_losses[event] = criterion(getattr(dataset, event + '_df_scaled_tensor_train')
                                              [indices], decoder_outputs[event])
        p_encoder_output = self.p_encoder(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        p_decoder_output = self.p_decoder(p_encoder_output)
        loss += criterion(dataset.patients_df_scaled_tensor_train[batch.indices_patients], p_decoder_output)
        # if loss != loss:
        #     for event in dataset.event_names:
        #         print(f' {event} {len(getattr(batch, "indices_" + event))}')

        all_losses['patients'] = criterion(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients], p_decoder_output)

        return loss, all_losses


class AutoEncoderAdaNet(Model):
    def __init__(self, model_specifics, device):
        super().__init__(device)
        self.size_embedding = model_specifics['size_embedding']
        self.num_targets = model_specifics['num_targets']
        self.encoders = {name: PaddedEncoderBis(model_specifics[name]['num_features'], model_specifics[name]['size_out'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                                model_specifics['dropout']).to(device) for name in model_specifics['event_names']}
        self.decoders = {name: DecoderBis(model_specifics['size_embedding'], model_specifics[name]['num_features'], model_specifics['num_layers_enc'],
                                          ).to(device) for name in model_specifics['event_names']}
        self.p_encoder = PaddedEncoderBis(model_specifics['patients']['num_features'], model_specifics['patients']['size_out'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                          model_specifics['dropout']).to(device)
        self.p_decoder = DecoderBis(model_specifics['size_embedding'], model_specifics['patients']['num_features'], model_specifics['num_layers_enc']
                                    ).to(device)
        self.LModule = LSTMModule(model_specifics['size_embedding'], model_specifics['device'],
                                  model_specifics['batch_first'], model_specifics['size_history'], model_specifics['num_layers']).to(device)
        # + 1 for time to prediction
        # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
        #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.PModule = PredModule(model_specifics['size_history'] + model_specifics['size_embedding'] +
                                  1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.task = model_specifics['task']
        self.batch_first = model_specifics['batch_first']

        self.parameters = list(self.p_encoder.parameters()) + list(self.p_decoder.parameters()) + \
            list(self.LModule.parameters()) + list(self.PModule.parameters())

        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())
            self.parameters += list(self.decoders[name].parameters())

    def train(self):

        for name in self.encoders:
            self.encoders[name].train()
            self.decoders[name].train()
        self.p_encoder.train()
        self.p_decoder.train()
        self.LModule.train()
        self.PModule.train()

    def eval(self):
        for name in self.encoders:
            self.encoders[name].eval()
            self.decoders[name].eval()
        self.p_encoder.eval()
        self.p_decoder.eval()
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, decoding_criterion, pred_criterion, batch):
        encoder_outputs = {}
        decoder_outputs = {}
        all_losses = {}
        decoding_loss = 0
        # apply encoders

        for event in dataset.event_names:
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
                decoder_outputs[event] = self.decoders[event](encoder_outputs[event])

                decoding_loss += decoding_criterion(getattr(dataset, event +
                                                    '_df_scaled_tensor_train')[indices], decoder_outputs[event])
                all_losses[event] = decoding_criterion(getattr(dataset, event + '_df_scaled_tensor_train')
                                                       [indices], decoder_outputs[event])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)

        p_encoder_output = self.p_encoder(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        p_decoder_output = self.p_decoder(p_encoder_output)
        decoding_loss += decoding_criterion(
            dataset.patients_df_scaled_tensor_train[batch.indices_patients], p_decoder_output)

        # for scaling of loss
        prediction_loss = 0
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype=torch.int64, device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v
                    combined = torch.zeros(size=(seq.sum(), self.size_embedding), device=self.device)
                    for index, event in enumerate(dataset.event_names):
                        mask = getattr(batch, event + '_masks')
                        combined[torch.broadcast_to(mask[patient][v], (len(mask[patient][v]), self.size_embedding))
                                 ] = encoder_outputs[event][indices[index]:indices[index] + seq[index]].flatten()
                    sequence.append(combined)
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                seq[visit_index], dataset.target_value_index]
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                  seq[visit_index], dataset.time_index]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                    seq[visit_index], dataset.target_index]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=self.batch_first)
            # compute the lengths of the sequences for each patient with available visit v
            lengths = batch.seq_lengths[v].sum(dim=1)[batch.available_visit_mask[:, v]].cpu()

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            history = hn[-1]
            # concat computed patient history with general information
            # general_info = self.p_encoder(
            #     dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]])
            general_info = p_encoder_output[batch.available_visit_mask[:, v]]

            pred_input = torch.cat((general_info, history, time_to_targets), dim=1)
            # pred_input = torch.cat(
            #     (dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]], history, time_to_targets), dim=1)
            # apply prediction module
            out = self.PModule(pred_input)
            # compute loss
            if self.task == 'classification':
                targets = target_categories
            else:
                targets = target_values
            prediction_loss += pred_criterion(out, targets)
            num_targets += len(targets)
            # # compute other training metrics
            # if self.task == 'classification':
            #     predictions = torch.tensor([torch.argmax(elem) for elem in out], device=self.device)
            # else:
            #     predictions = out
            #metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        prediction_loss = prediction_loss / num_targets

        return decoding_loss, prediction_loss

    def apply(self, dataset, patient_id):

        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            predictions = torch.empty(size=(len(dataset[patient_id].visits) -
                                            dataset.min_num_visits + 1, self.num_targets), device=self.device)
            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]

            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]
            all_history = torch.empty(size=(max_num_visits - dataset.min_num_visits +
                                      1, self.LModule.hidden_size), device=self.device)
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # create combined ordered list of visit/medication/events up to v
                combined = torch.zeros(size=(seq_lengths[visit].sum(), self.size_embedding), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    mask = getattr(dataset.masks, event + '_masks')[patient_mask_index]
                    combined[torch.broadcast_to(mask[visit], (len(mask[visit]), self.size_embedding))
                             ] = encoder_outputs[event][:seq_lengths[visit][index]].flatten()
                time_to_target = torch.tensor(
                    [[dataset[patient_id].targets_df_tensor[visit - 1, 0]]], device=self.device)
                # apply lstm
                padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
                lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
                pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
                output, (hn, cn) = self.LModule(pack_padded)
                history = hn[-1]
                all_history[visit, :] = history
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat((general_info, history, time_to_target), dim=1)
                # pred_input = torch.cat(
                #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
                # apply prediction module
                predictions[visit - dataset.min_num_visits] = self.PModule(pred_input)
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, all_history


class Othernet(Model):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.size_embedding = config['size_embedding']
        self.size_event_history = config['size_history']
        self.num_targets = config['num_targets']

        self.encoders = {name: PaddedEncoder(config[name]['num_features'], config[name]['size_out'], config['size_embedding'], config['num_layers_enc'],
                                                config['hidden_enc'], config['dropout']).to(device) for name in config['event_names']}

        self.p_encoder = PaddedEncoder(config['patients']['num_features'], config['patients']['size_out'], config['size_embedding'], config['num_layers_enc'],
                                        config['hidden_enc'], config['dropout']).to(device)
        
        self.lstm_modules = {name: LSTMModule(config['size_embedding'], config['device'],
                                                config['batch_first'], config['size_history'], config['num_layers']).to(device) for name in config['event_names']}
        # + 1 for time to prediction
        # self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
        #                           1, model_specifics['num_targets'], model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.PModule = PredModule(config['size_history'] * len(config['event_names']) + config['size_embedding'] +
                                  1, config['num_targets'], config['num_layers_pred'], config['hidden_pred'], config['dropout']).to(device)
        self.batch_first = config['batch_first']

        self.parameters = list(self.PModule.parameters()) + list(self.p_encoder.parameters())
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
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, criterion, batch):
        loss = 0
        encoder_outputs = {}
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype=torch.int64, device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            combined = torch.zeros(size=(len(batch.current_indices), len(dataset.event_names) * self.size_event_history), device=self.device)
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v

                    for index, event in enumerate(dataset.event_names):
                        #mask = getattr(batch, event + '_masks')
                        # combined[torch.broadcast_to(mask[patient][v], (len(mask[patient][v]), self.size_embedding))
                        #          ] = encoder_outputs[event][indices[index]:indices[index] + seq[index]].flatten()
                        if seq[index]> 0:
                            combined[patient, index * self.size_event_history:index * self.size_event_history + self.size_event_history] = self.lstm_modules[event](
                                encoder_outputs[event][indices[index]:indices[index] + seq[index]].flatten())[1][0][-1]
                        else:
                            continue
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                seq[visit_index], dataset.target_value_index]
                    #TODO change
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                  seq[visit_index], dataset.time_index]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                    seq[visit_index], dataset.target_index]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]

            general_info = patient_encoding[batch.available_visit_mask[:, v]]

            pred_input = torch.cat((general_info, combined, time_to_targets), dim=1)
            # pred_input = torch.cat(
            #     (dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]], history, time_to_targets), dim=1)
            # apply prediction module
            out = self.PModule(pred_input)

            targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)
            # # compute other training metrics
            # if self.task == 'classification':
            #     predictions = torch.tensor([torch.argmax(elem) for elem in out], device=self.device)
            # else:
            #     predictions = out
            #metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets

    def apply(self, dataset, patient_id):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            
            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            predictions = torch.empty(size=(torch.sum(available_visit_mask == True).item(), self.num_targets), device=self.device)
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]
            all_history = torch.empty(size=(max_num_visits - dataset.min_num_visits +
                                    1, self.LModule.hidden_size), device=self.device)
            
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(),1), dtype=torch.int64, device=self.device)
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1), device=self.device)

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v
                combined = torch.zeros(size=(seq_lengths[visit].sum(), self.size_embedding), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    mask = getattr(dataset.masks, event + '_masks')[patient_mask_index]
                    combined[torch.broadcast_to(mask[visit], (len(mask[visit]), self.size_embedding))
                            ] = encoder_outputs[event][:seq_lengths[visit][index]].flatten()

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0], dataset.target_value_index]
                #TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                    dataset.time_index]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                        dataset.target_index]
                # apply lstm
                padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
                lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
                pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                    padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
                output, (hn, cn) = self.LModule(pack_padded)
                history = hn[-1]
                all_history[visit, :] = history
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat((general_info, history, time_to_targets[index_target]), dim=1)
                # pred_input = torch.cat(
                #     (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
                # apply prediction module
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, all_history, target_values, time_to_targets, target_categories


class Othernet(Model):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.size_embedding = config['size_embedding']
        self.combined_history_size = (sum([config[name]['size_history'] for name in config['event_names']]), np.cumsum([
                                      0] + [config[name]['size_history'] for name in config['event_names']]))
        self.config = config

        self.encoders = {name: Encoder(config[name]['num_features'], config[name]['size_out'], config['num_layers_enc'],
                                       config['hidden_enc'], config['dropout']).to(device) for name in config['event_names']}

        self.p_encoder = Encoder(config['patients']['num_features'], config['patients']['size_out'], config['num_layers_enc'],
                                 config['hidden_enc'], config['dropout']).to(device)

        self.lstm_modules = {name: LSTMModulebis(config[name]['size_out'], config['device'],
                                                 config['batch_first'], config[name]['size_history'], config['num_layers']).to(device) for name in config['event_names']}
        # + 1 for time to prediction

        self.PModule = PredModule(self.combined_history_size[0] + config['patients']['size_out'] +
                                  1, 1, config['num_layers_pred'], config['hidden_pred'], config['dropout']).to(device)
        self.batch_first = config['batch_first']

        self.parameters = list(self.PModule.parameters()) + list(self.p_encoder.parameters())
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

    def apply_and_get_loss(self, dataset, criterion, batch):
        loss = 0
        encoder_outputs = {}
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            #sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            indices_lstm = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets caetgories
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype=torch.int64, device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)

            index_target = 0
            debug_index_target = None
            # to contain encoder outputs. For each event, each patient is represented by a vector of shape max_num_events * embedding size
            # if a patient has less events, the last rows are zero padded
            combined = {event: torch.zeros(size=(len(batch.current_indices), batch.seq_lengths[v][:, index].max().item(
            ), self.config[event]['size_out']), device=self.device) for index, event in enumerate(dataset.event_names)}

            # combined representation (before prediction)
            combined_lstm = torch.zeros(size=(len(batch.current_indices),
                                        self.combined_history_size[0]), device=self.device)
            # have all encoder outputs
            #use sequ lengths as lengths

            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.available_visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication/events up to v

                    for index, event in enumerate(dataset.event_names):
                        if seq[index] > 0:
                            combined[event][patient, :seq[index],
                                            :] = encoder_outputs[event][indices[index]:indices[index] + seq[index]]
                        else:
                            continue
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                seq[visit_index], dataset.target_value_index]
                    #TODO change
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                  seq[visit_index], dataset.time_index]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices[visit_index] +
                                                                                                                    seq[visit_index], dataset.target_index]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                    index_target += 1
                # update the indices to select from in the tensors
                for index, event in enumerate(dataset.event_names):
                    indices[index] += batch.total_num[patient, index]
                    indices_lstm[index] += 1
            for index, event in enumerate(dataset.event_names):
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                if len(patients) > 0:
                    combined_lstm[batch.available_visit_mask[:, v], self.combined_history_size[1][index]:self.combined_history_size[1]
                                  [index + 1]] = self.lstm_modules[event](combined[event][batch.available_visit_mask[:, v]])[1][0][-1]
            general_info = patient_encoding[batch.available_visit_mask[:, v]]
            pred_input = torch.cat(
                (general_info, combined_lstm[batch.available_visit_mask[:, v]], time_to_targets), dim=1)

            # apply prediction module
            out = self.PModule(pred_input)

            targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets

    def apply(self, dataset, patient_id):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            predictions = torch.empty(size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), dtype=torch.int64, device=self.device)
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1), device=self.device)

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v

                combined = {event: torch.zeros(size=(1, seq_lengths[visit][index].item(
                ), self.config[event]['size_out']), device=self.device) for index, event in enumerate(dataset.event_names)}

                combined_lstm = torch.zeros(size=(1, self.combined_history_size[0]), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index] > 0:

                        combined[event][:seq_lengths[visit][index],
                                        :] = encoder_outputs[event][:seq_lengths[visit][index]]

                    else:
                        continue

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                    dataset.target_value_index]
                #TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                      dataset.time_index]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                        dataset.target_index]
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index].item() > 0:
                        combined_lstm[0, self.combined_history_size[1][index]:self.combined_history_size[1]
                                      [index + 1]] = self.lstm_modules[event](combined[event])[1][0][-1]
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat(
                    (general_info, combined_lstm[available_visit_mask[visit]].reshape(1, combined_lstm[available_visit_mask[visit]].shape[2]), time_to_targets[index_target]), dim=1)
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, 'nothing', target_values, time_to_targets, target_categories


class Othernet2(Model):
    def __init__(self, config, device):
        super().__init__(config, device)

        self.size_embedding = config['size_embedding']
        self.combined_history_size = (sum([config[name]['size_history'] for name in config['event_names']]), np.cumsum([
                                      0] + [config[name]['size_history'] for name in config['event_names']]))
        self.config = config

        self.encoders = {name: Encoder(config[name]['num_features'], config[name]['size_out'], config['num_layers_enc'],
                                       config['hidden_enc'], config['dropout']).to(device) for name in config['event_names']}

        self.p_encoder = Encoder(config['patients']['num_features'], config['patients']['size_out'], config['num_layers_enc'],
                                 config['hidden_enc'], config['dropout']).to(device)

        self.lstm_modules = {name: LSTMModulebis(config[name]['size_out'], config['device'],
                                                 config['batch_first'], config[name]['size_history'], config['num_layers']).to(device) for name in config['event_names']}
        # + 1 for time to prediction

        self.PModule = PredModule(self.combined_history_size[0] + config['patients']['size_out'] +
                                  1, 1, config['num_layers_pred'], config['hidden_pred'], config['dropout']).to(device)
        self.batch_first = config['batch_first']

        self.parameters = list(self.PModule.parameters()) + list(self.p_encoder.parameters())
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

    def apply_and_get_loss(self, dataset, criterion, batch):
        loss = 0
        encoder_outputs = {}
        # apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, 'indices_' + event)
            if len(indices) > 0:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset, event + '_df_scaled_tensor_train')[indices])
            else:
                encoder_outputs[event] = torch.tensor([], device=self.device)
        patient_encoding = self.p_encoder(dataset.patients_df_scaled_tensor_train[batch.indices_patients])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # continue if this visit shouldn't be predicted for any patient
            if torch.sum(batch.available_visit_mask[:, v] == True).item() == 0:
                continue
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            #sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            indices_lstm = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device=self.device)
            visit_index = dataset.event_names.index('a_visit')

            index_target = 0
            debug_index_target = None
            # to contain encoder outputs. For each event, each patient is represented by a vector of shape max_num_events * embedding size
            # if a patient has less events, the last rows are zero padded
            combined = {event: torch.zeros(size=(len(batch.current_indices), batch.seq_lengths[v][:, index].max().item(
            ), self.config[event]['size_out']), device=self.device) for index, event in enumerate(dataset.event_names)}

            # to slice encoder outputs
            indices_to_keep = {event: [] for event in dataset.event_names}
            indices_combined = {event: [] for event in dataset.event_names}
            indices_targets = [batch.seq_lengths[v][p, visit_index] + batch.total_num[:p, visit_index].sum()
                               for p in range(len(batch.seq_lengths[v]))]
            for index, event in enumerate(dataset.event_names):
                indices_to_keep[event].extend(np.arange(0, batch.seq_lengths[v, 0, index], 1))
                indices_combined[event].extend(np.arange(0, batch.seq_lengths[v, 0, index], 1))

                for i in range(1, len(batch.seq_lengths[v])):
                    indices_to_keep[event].extend(np.arange(batch.total_num[:i, index].sum(
                    ), batch.total_num[:i, index].sum() + batch.seq_lengths[v, i, index], 1))
                    indices_combined[event].extend(np.arange(batch.seq_lengths[v][:, index].max().item(
                    ) * i, batch.seq_lengths[v][:, index].max().item() * i + batch.seq_lengths[v, i, index], 1))


#                 print(indices_to_keep[event])
#                 print(indices_combined[event])
#                 print(combined[event].shape)
#                 print(combined[event])
#                 print(encoder_outputs[event].shape)
#                 print(encoder_outputs[event])
                if len(indices_to_keep[event]) > 0:
                    combined[event].view((combined[event].shape[0] * combined[event].shape[1], combined[event].shape[2])
                                         )[indices_combined[event]] = encoder_outputs[event][indices_to_keep[event]]
            # targets
            target_values = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices_targets,
                                                                                          dataset.target_value_index]
            target_values = target_values.reshape(len(target_values), 1)[batch.available_visit_mask[:, v]]
            time_to_targets = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices_targets, dataset.time_index]
            time_to_targets = time_to_targets.reshape(len(time_to_targets), 1)[batch.available_visit_mask[:, v]]
            target_categories = dataset.targets_df_scaled_tensor_train[batch.indices_targets][indices_targets, dataset.target_index]
            target_categories = target_categories.reshape(len(target_categories), 1)[batch.available_visit_mask[:, v]]
            #lengths = {event: batch.seq_lengths[v,index, :] for index, event in enumerate(dataset.event_names)}
            # combined representation (before prediction)
            combined_lstm = torch.zeros(size=(len(batch.current_indices),
                                        self.combined_history_size[0]), device=self.device)

            for index, event in enumerate(dataset.event_names):
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                if len(patients) > 0:
                    # "preprocessing" to apply lstm
                    #                     padded_sequence = torch.nn.utils.rnn.pad_sequence(combined[event][batch.available_visit_mask[:, v]], batch_first=self.batch_first)
                    #                     # compute the lengths of the sequences for each patient with available visit v
                    #                     lengths = lengths[event][batch.available_visit_mask[:, v]].cpu()

                    #                     pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                    #                         padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

                    combined_lstm[batch.available_visit_mask[:, v], self.combined_history_size[1][index]:self.combined_history_size[1]
                                  [index + 1]] = self.lstm_modules[event](combined[event][batch.available_visit_mask[:, v]])[1][0][-1]
            general_info = patient_encoding[batch.available_visit_mask[:, v]]

            pred_input = torch.cat(
                (general_info, combined_lstm[batch.available_visit_mask[:, v]], time_to_targets), dim=1)

            # apply prediction module
            out = self.PModule(pred_input)

            targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets

    def apply(self, dataset, patient_id):
        with torch.no_grad():
            # method to directly apply the model to a single patient
            patient_mask_index = dataset.mapping_for_masks[patient_id]
            encoder_outputs = {}
            for event in dataset.event_names:
                encoder_outputs[event] = self.encoders[event](
                    getattr(dataset[patient_id], event + '_df_tensor').to(self.device))

            seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]
            available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
            predictions = torch.empty(size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            max_num_visits = dataset.masks.num_visits[patient_mask_index]
            total_num = dataset.masks.total_num[patient_mask_index]

            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), device=self.device)
            # targets categories
            target_categories = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1), dtype=torch.int64, device=self.device)
            time_to_targets = torch.empty(
                size=(torch.sum(available_visit_mask == True).item(), 1, 1), device=self.device)

            index_target = 0
            for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
                # continue if this visit shouldn't be predicted for any patient
                if torch.sum(available_visit_mask[visit] == True).item() == 0:
                    continue
                # create combined ordered list of visit/medication/events up to v

                combined = {event: torch.zeros(size=(1, seq_lengths[visit][index].item(
                ), self.config[event]['size_out']), device=self.device) for index, event in enumerate(dataset.event_names)}

                combined_lstm = torch.zeros(size=(1, self.combined_history_size[0]), device=self.device)
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index] > 0:

                        combined[event][:seq_lengths[visit][index],
                                        :] = encoder_outputs[event][:seq_lengths[visit][index]]

                    else:
                        continue

                # targets (values)
                target_values[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                    dataset.target_value_index]
                #TODO change

                time_to_targets[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                      dataset.time_index]

                target_categories[index_target] = dataset[patient_id].targets_df_tensor[seq_lengths[visit, 0],
                                                                                        dataset.target_index]
                for index, event in enumerate(dataset.event_names):
                    if seq_lengths[visit][index].item() > 0:
                        combined_lstm[0, self.combined_history_size[1][index]:self.combined_history_size[1]
                                      [index + 1]] = self.lstm_modules[event](combined[event])[1][0][-1]
                # concat computed patient history with general information
                general_info = self.p_encoder(
                    dataset[patient_id].patients_df_tensor.to(self.device))
                pred_input = torch.cat(
                    (general_info, combined_lstm[available_visit_mask[visit]].reshape(1, combined_lstm[available_visit_mask[visit]].shape[2]), time_to_targets[index_target]), dim=1)
                predictions[index_target] = self.PModule(pred_input)
                index_target += 1
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, 'nothing', target_values, time_to_targets, target_categories



