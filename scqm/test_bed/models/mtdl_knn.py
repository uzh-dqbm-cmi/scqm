import torch
import copy
import pandas as pd
import numpy as np

from scqm.test_bed.fake_scqm import get_df_dict
from scqm.custom_library.preprocessing.select_features import extract_adanet_features
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.models.model import Model
from scqm.custom_library.models.modules.encoders import EventEncoder

from scqm.custom_library.models.modules.lstms import LstmEventSpecific
from scqm.custom_library.models.modules.predictions import PredModule

from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch


class OthernetOptimized(Model):
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

            visit_index = dataset.event_names.index("a_visit")

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
            # to index encoder output
            indices_to_keep = {event: [] for event in dataset.event_names}
            # to index combined output
            indices_combined = {event: [] for event in dataset.event_names}
            indices_targets = [
                batch.seq_lengths[v][p, visit_index]
                + batch.total_num[:p, visit_index].sum()
                for p in range(len(batch.seq_lengths[v]))
            ]
            for index, event in enumerate(dataset.event_names):
                # first patient
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

                if len(indices_to_keep[event]) > 0:
                    # (shape num_patients * max_num_event) x encoder_output_size
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
                batch.available_target_mask[:, v]
            ]
            time_to_targets = dataset.targets_df_scaled_tensor_train[
                batch.indices_targets
            ][indices_targets, dataset.time_index]
            time_to_targets = time_to_targets.reshape(len(time_to_targets), 1)[
                batch.available_target_mask[:, v]
            ]
            target_categories = dataset.targets_df_scaled_tensor_train[
                batch.indices_targets
            ][indices_targets, dataset.target_index]
            target_categories = target_categories.reshape(len(target_categories), 1)[
                batch.available_target_mask[:, v]
            ]
            # lengths = {event: batch.seq_lengths[v,index, :] for index, event in enumerate(dataset.event_names)}
            # combined representation (before prediction)
            combined_lstm = torch.zeros(
                size=(len(batch.current_indices), self.combined_history_size[0]),
                device=self.device,
            )

            for index, event in enumerate(dataset.event_names):
                patients = torch.nonzero(batch.seq_lengths[v][:, index]).flatten()
                patients = [
                    elem.item()
                    for elem in patients
                    if batch.available_target_mask[elem, v]
                ]
                if len(patients) > 0:
                    # "preprocessing" to apply lstm
                    #                     padded_sequence = torch.nn.utils.rnn.pad_sequence(combined[event][batch.available_target_mask[:, v]], batch_first=self.batch_first)
                    #                     # compute the lengths of the sequences for each patient with available visit v
                    #                     lengths = lengths[event][batch.available_target_mask[:, v]].cpu()

                    #                     pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                    #                         padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)
                    lengths = batch.seq_lengths[v, patients, index].cpu()

                    pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                        combined[event][patients, :, :],
                        batch_first=self.batch_first,
                        lengths=lengths,
                        enforce_sorted=False,
                    )

                    combined_lstm[
                        patients,
                        self.combined_history_size[1][
                            index
                        ] : self.combined_history_size[1][index + 1],
                    ] = self.lstm_modules[event](pack_padded_sequence)[1][0][-1]
            general_info = patient_encoding[batch.available_target_mask[:, v]]

            pred_input = torch.cat(
                (
                    general_info,
                    combined_lstm[batch.available_target_mask[:, v]],
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


if __name__ == "__main__":
    # create fake data
    df_dict = get_df_dict(num_patients=100)

    real_data = False
    df_dict_processed = copy.deepcopy(df_dict)
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(
            pd.to_datetime
        )
    (
        patients_df,
        medications_df,
        visits_df,
        targets_df,
        _,
        _,
        haq_df,
        _,
    ) = extract_adanet_features(
        df_dict_processed,
        transform_meds=True,
        das28=True,
        only_meds=True,
        joint_df=False,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": patients_df,
        "med": medications_df,
        "targets": targets_df,
        "haq": haq_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = Dataset(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        "das28_increase",
        ["a_visit", "med", "haq"],
        min_num_targets,
    )
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks()
    dataset.create_dfs()
    # prepare for training
    dataset.transform_to_numeric_adanet(real_data)
    partition = DataPartition(dataset, k=3)
    fold = int(0)
    partition.set_current_fold(fold)
    model_specifics = {}
