import torch
from scqm.custom_library.modules import VisitEncoder, MedicationEncoder, LSTMModule, PredModule

#base model class


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
        self.VEncoder = VisitEncoder(model_specifics['num_visit_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                     model_specifics['hidden_enc'], model_specifics['dropout']).to(device)
        self.MEncoder = MedicationEncoder(model_specifics['num_medications_features'], model_specifics['size_embedding'],
                                          model_specifics['num_layers_enc'], model_specifics['hidden_enc'], model_specifics['dropout']).to(device)
        self.LModule = LSTMModule(model_specifics['size_embedding'], model_specifics['device'],
                                  model_specifics['batch_first'], model_specifics['size_history'], model_specifics['num_layers']).to(device)
        # + 1 for time to prediction
        self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
                                  1, model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.task = model_specifics['task']
        self.batch_first = model_specifics['batch_first']
        self.parameters = list(self.VEncoder.parameters()) + list(self.MEncoder.parameters()) + \
            list(self.LModule.parameters()) + list(self.PModule.parameters())

    def train(self):
        self.VEncoder.train()
        self.MEncoder.train()
        self.LModule.train()
        self.PModule.train()

    def eval(self):
        self.VEncoder.eval()
        self.MEncoder.eval()
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, criterion, batch, metrics):
        loss = 0
        o_v, o_m = self.VEncoder(dataset.visits_df_scaled_tensor_train[batch.indices_v]), self.MEncoder(
            dataset.medications_df_scaled_tensor_train[batch.indices_m])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            index_visits = 0
            index_medications = 0
            # targets (values)
            target_values = torch.empty(size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets (categories : 0 if <= 2.6 else 1)
            target_categories = torch.empty(
                size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # delta t
            time_to_targets = torch.empty(
                size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication events up to v
                    combined = torch.zeros_like(torch.cat([o_v[index_visits:index_visits + seq[0]],
                                                           o_m[index_medications:index_medications + seq[1]]]))
                    combined[batch.masks[patient][v]] = o_v[index_visits:index_visits + seq[0]].flatten()
                    combined[~batch.masks[patient][v]] = o_m[index_medications:index_medications + seq[1]].flatten()

                    sequence.append(combined)
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 1]
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 0]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 2]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                        #print(f'visit info {tensor_v[index_visits:index_visits + seq[0]]} \n medication info {tensor_m[index_medications:index_medications + seq[1]]}')
                    index_target += 1
                # update the indices to select from in the tensors
                index_visits += batch.total_num[patient, 0]
                index_medications += batch.total_num[patient, 1]
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=self.batch_first)
            # compute the lengths of the sequences for each patient with available visit v
            lengths = batch.seq_lengths[v].sum(dim=1)[batch.visit_mask[:, v]]

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            history = hn[-1]
            # concat computed patient history with general information
            pred_input = torch.cat(
                (dataset.patients_df_scaled_tensor_train[batch.indices_p][batch.visit_mask[:, v]], history, time_to_targets), dim=1)
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
                predictions = torch.tensor([1 if elem >= 0.5 else 0 for elem in out], device=self.device)
            else:
                predictions = out
            metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target].item()}')

        return loss / num_targets, metrics

    def apply(self, dataset, patient_id):
        # method to directly apply the model to a single patient

        o_v, o_m = self.VEncoder(dataset[patient_id].visits_df_tensor), self.MEncoder(
            dataset[patient_id].medications_df_tensor)

        predictions = torch.empty(size=(len(dataset[patient_id].visits) -
                                  dataset.min_num_visits + 1, 1), device=self.device)
        for visit in range(dataset.min_num_visits, len(dataset[patient_id].visits) + 1):
            # create combined ordered list of visit/medication events up to v
            num_visits, num_meds, _, cropped_timeline_mask, _ = dataset[patient_id].get_cropped_timeline(
                visit)
            mask = torch.broadcast_to(torch.tensor([[tuple_[0]] for tuple_ in cropped_timeline_mask]),
                                      (len(cropped_timeline_mask), self.VEncoder.size_embedding))
            combined = torch.zeros_like(torch.cat([o_v[:num_visits],
                                                   o_m[:num_meds]]))

            combined[mask] = o_v[:num_visits].flatten()
            combined[~mask] = o_m[:num_meds].flatten()
            time_to_target = torch.tensor([[dataset[patient_id].targets_df_tensor[visit - 1, 0]]])
            # apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence([combined], batch_first=self.batch_first)
            pack_padded = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, batch_first=self.batch_first, lengths=torch.tensor([
                                                                  num_visits + num_meds], dtype=torch.int64), enforce_sorted=False)
            output, (hn, cn) = self.LModule(pack_padded)
            history = hn[-1]
            # concat computed patient history with general information
            pred_input = torch.cat(
                (dataset[patient_id].patients_df_tensor, history, time_to_target), dim=1)
            # apply prediction module
            predictions[visit - dataset.min_num_visits] = self.PModule(pred_input)

        predicted_categories = [1 if elem >= 0.5 else 0 for elem in predictions]

        return predictions, predicted_categories
