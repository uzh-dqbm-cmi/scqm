import torch
from modules import Encoder, LSTMModule, PredModule

# base model class


class Model:
    def __init__(self, device):
        self.device = device

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError


class Adaptivenet(Model):
    def __init__(self, model_specifics, device):
        super().__init__(device)
        self.size_embedding = model_specifics['size_embedding']
        self.num_targets = model_specifics['num_targets']
        self.encoders = {name: Encoder(model_specifics[name]['num_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                       model_specifics['hidden_enc'], model_specifics['dropout']).to(device) for name in model_specifics['event_names']}


        self.LModule = LSTMModule(model_specifics['size_embedding'], model_specifics['device'],
                                  model_specifics['batch_first'], model_specifics['size_history'], model_specifics['num_layers']).to(device)
        # + 1 for time to prediction
        self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
                                  1, model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.task = model_specifics['task']
        self.batch_first = model_specifics['batch_first']

        self.parameters = list(self.LModule.parameters()) + list(self.PModule.parameters())

        for name in self.encoders:
            self.parameters += list(self.encoders[name].parameters())

    def train(self):

        for name in self.encoders:
            self.encoders[name].train()
        self.LModule.train()
        self.PModule.train()

    def eval(self):

        for name in self.encoders:
            self.encoders[name].eval()
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, criterion, batch, metrics):
        loss = 0
        encoder_outputs = {}
        #apply encoders
        for event in dataset.event_names:
            indices = getattr(batch, 'indices_'+event)
            encoder_outputs[event] = self.encoders[event](getattr(dataset, event + '_df_scaled_tensor_train')[indices])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            indices = torch.zeros((len(dataset.event_names),), dtype=torch.int32, device = self.device)
            visit_index = dataset.event_names.index('a_visit')
            # targets (values)
            target_values = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets (categories : 0 if <= 2.6 else 1)
            target_categories = torch.empty(
                size=(torch.sum(batch.available_visit_mask[:, v] == True).item(),), dtype = torch.int64, device=self.device)
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
                    combined = torch.zeros(size=(seq.sum(), self.size_embedding), device = self.device)
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
            pred_input = torch.cat(
                (dataset.patients_df_scaled_tensor_train[batch.indices_patients][batch.available_visit_mask[:, v]], history, time_to_targets), dim=1)
            # apply prediction module
            out = self.PModule(pred_input)
            # compute loss
            if self.task == 'classification':
                targets = target_categories
            else:
                targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)
            # compute other training metrics
            if self.task == 'classification':
                predictions = torch.tensor([torch.argmax(elem) for elem in out], device=self.device)
            else:
                predictions = out
            metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target]}')

        return loss / num_targets, metrics

    def apply(self, dataset, patient_id):


        # method to directly apply the model to a single patient
        patient_mask_index = dataset.mapping_for_masks[patient_id]
        encoder_outputs = {}
        for event in dataset.event_names:
            encoder_outputs[event] = self.encoders[event](getattr(dataset[patient_id], event + '_df_tensor'))

        predictions = torch.empty(size=(len(dataset[patient_id].visits) -
                                        dataset.min_num_visits + 1, self.num_targets), device=self.device)
        seq_lengths = dataset.masks.seq_lengths[:, patient_mask_index, :]

        available_visit_mask = dataset.masks.available_visit_mask[patient_mask_index]
        max_num_visits = dataset.masks.num_visits[patient_mask_index]
        total_num = dataset.masks.total_num[patient_mask_index]
        for visit in range(0, max_num_visits - dataset.min_num_visits + 1):
            # create combined ordered list of visit/medication/events up to v
            combined = torch.zeros(size=(seq_lengths[visit].sum(), self.size_embedding), device=self.device)
            for index, event in enumerate(dataset.event_names):
                mask = getattr(dataset.masks, event + '_masks')[patient_mask_index]
                combined[torch.broadcast_to(mask[visit], (len(mask[visit]), self.size_embedding))
                        ] = encoder_outputs[event][:seq_lengths[visit][index]].flatten()
            time_to_target = torch.tensor([[dataset[patient_id].targets_df_tensor[visit - 1, 0]]], device=self.device)
            # apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
            lengths = seq_lengths[visit].sum()[available_visit_mask[visit]].cpu()
            pack_padded = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
            output, (hn, cn) = self.LModule(pack_padded)
            history = hn[-1]
            # concat computed patient history with general information
            pred_input = torch.cat(
                (dataset[patient_id].patients_df_tensor.to(self.device), history, time_to_target), dim=1)
            # apply prediction module
            predictions[visit - dataset.min_num_visits] = self.PModule(pred_input)

        predicted_categories = torch.tensor([torch.argmax(elem) for elem in predictions], device=self.device)

        return predictions, predicted_categories


