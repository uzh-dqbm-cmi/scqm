from scqm.custom_library.utils import set_seeds
import torch
import pickle
from scqm.custom_library.plot.shap.utils import *
from scqm.custom_library.models.MLP import MLP
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    set_seeds(0)
    print("start")
    target_name = "das283bsr_score"
    to_save = {"target_name": target_name}

    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_10_11.pickle", "rb"
    ) as f:
        cv = pickle.load(f)
    dataset = cv.dataset
    partition = cv.partition
    for fold in range(5):
        print(fold)
        to_save[fold] = {}
        if target_name == "asdas_score":
            patients_train = partition.partitions_train_asdas[fold]
            patients_valid = partition.partitions_test_asdas[fold]
        elif target_name == "das283bsr_score":
            patients_train = partition.partitions_train_das28[fold]
            patients_valid = partition.partitions_test_das28[fold]
        patients_test = [
            p
            for p in dataset.test_ids
            if dataset[p].target_name in ["both", target_name]
        ]
        patients_all = patients_train + patients_valid + patients_test
        features, targets = prepare_features(dataset, patients_all, target_name)
        df, X_train, y_train, X_valid, y_valid, X_test, y_test = dfs_as_numeric(
            features, targets, patients_train, patients_valid, patients_test
        )
        (
            X_train_scaled,
            X_valid_scaled,
            X_test_scaled,
            y_train_scaled,
            y_valid_scaled,
            y_test_scaled,
            feat_min_,
            feat_max_,
            targ_min_,
            targ_max_,
        ) = ml_prepro(X_train, y_train, X_valid, y_valid, X_test, y_test)

        to_save[fold]["df"] = df
        to_save[fold]["targ_min"] = targ_min_
        to_save[fold]["targ_ax"] = targ_max_
        to_save[fold]["targ_min"] = targ_min_
        to_save[fold]["X_train_scaled"] = X_train_scaled
        to_save[fold]["X_test_scaled"] = X_test_scaled
        to_save[fold]["y_test"] = y_test

        input_size = X_train.shape[1]
        if target_name == "das283bsr_score":
            config = {
                "input_size": input_size,
                "output_size": 1,
                "num_hidden": 10,
                "hidden_size": 100,
            }
        elif target_name == "asdas_score":
            config = {
                "input_size": input_size,
                "output_size": 1,
                "num_hidden": 3,
                "hidden_size": 100,
            }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MLP(config, device)
        batchsize = 300
        n_epochs = 400
        lr = 1e-3
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, device=device, dtype=torch.float32),
            torch.tensor(y_train_scaled, device=device, dtype=torch.float32),
        )
        valid_dataset = TensorDataset(
            torch.tensor(X_valid_scaled, device=device, dtype=torch.float32),
            torch.tensor(y_valid_scaled, device=device, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batchsize)
        valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batchsize)
        valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
        seed = 33
        losses_train = []
        losses_valid = []
        torch.manual_seed(seed)
        for epoch in range(n_epochs):
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                with torch.no_grad():
                    X_batch_valid, y_batch_valid = next(iter(valid_loader))
                    y_valid_pred = model(X_batch_valid)
                    loss_valid = criterion(y_valid_pred, y_batch_valid)
                    print(f"train loss : {loss} loss valid : {loss_valid}")
                    losses_train.append(loss.item())
                    losses_valid.append(loss_valid.item())
        to_save[fold]["model"] = model
        to_save[fold]["losses_train"] = losses_train
        to_save[fold]["losses_valid"] = losses_valid

    with open(
        "/cluster/work/medinfmk/scqm/tmp/baselines/shap_das28.pickle",
        "wb",
    ) as handle:
        pickle.dump(to_save, handle)
