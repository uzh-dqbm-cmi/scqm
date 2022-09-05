import matplotlib.pyplot as plt
import random
import numpy as np


def plot_patient_predictions(
    dataset, result_df, target_name, plot_time=True, patient_id=None
):
    df = result_df.copy()
    if target_name == "das283bsr_score":
        patients = [
            patient
            for patient in dataset.test_ids
            if dataset[patient].target_name == "das283bsr_score"
        ]
        ylabel = "Das28"
    elif target_name == "asdas_score":
        patients = [
            patient
            for patient in dataset.test_ids
            if dataset[patient].target_name == "asdas_score"
        ]
        ylabel = "asdas"
    if patient_id is None:
        p = random.sample(patients, 1)[0]
    else:
        p = patient_id
    plt.figure(figsize=(7, 5))
    plt.grid(visible=True, linewidth=2)
    plt.title("Predictions vs true values \n single patient")
    if plot_time:
        targets = df[df.patient_id == p].targets
        predictions = df[df.patient_id == p].predictions
        dates = df[df.patient_id == p].prediction_dates
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(dates, targets, "--o", label="true values", c="blue")
        plt.plot(dates, predictions, "--o", label="predictions", c="darkgreen")
        # plt.xticks(range(1, len(targets)+1))
        plt.xlabel("Date", fontsize=15)
        plt.ylim((0, 10))
        plt.ylabel(ylabel, fontsize=15)
        plt.legend()
    else:
        targets = df[df.patient_id == p].targets
        predictions = df[df.patient_id == p].predictions
        plt.plot(
            range(1, len(targets) + 1), targets, "--o", label="true values", c="blue"
        )
        plt.plot(
            range(1, len(targets) + 1),
            predictions,
            "--o",
            label="predictions",
            c="darkgreen",
        )
        plt.xticks(range(1, len(targets) + 1))
        plt.xlabel("Predicted visit", fontsize=15)
        plt.ylim((0, 10))
        plt.ylabel(ylabel, fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend()

    return df[df.patient_id == p]


def error_analysis(df, n_bins=15):
    tmp = df.copy()
    tmp["absolute_error"] = abs(df.predictions - df.targets)
    #     plt.figure()
    #     plt.scatter(tmp.targets, tmp.absolute_error)
    #     plt.title('Target value vs MAE')
    #     plt.ylabel('MAE')
    #     plt.xlabel('target value')
    plt.figure()
    plt.grid(visible=True)
    (freq, bins, _) = plt.hist(
        x=df["targets"],
        bins=n_bins,
        color="#0504aa",
        alpha=0.5,
        rwidth=0.85,
        label="targets",
    )
    plt.xlabel("value", fontsize=14)
    plt.ylabel("frequency", fontsize=14)
    plt.title("Frequency of target label", fontsize=14)
    indices_sorted = np.argsort(freq)
    bins_sorted = [
        (bins[index - 1], bins[index]) if index > 0 else (0, bins[index])
        for index in indices_sorted
    ]
    freq_sorted = [freq[index] for index in indices_sorted]
    tmp["bin"] = np.nan
    tmp["freq_bin"] = np.nan
    for elem in range(len(tmp)):
        for index, bin_ in enumerate(bins_sorted):
            if (bin_[0] <= tmp["targets"].iloc[elem]) & (
                tmp["targets"].iloc[elem] < bin_[1]
            ):
                tmp["bin"].iloc[elem] = index
                tmp["freq_bin"].iloc[elem] = freq_sorted[index]
    plt.figure()
    plt.grid(visible=True)
    plt.plot(
        tmp.groupby("freq_bin").apply(
            lambda x: sum((x.targets - x.predictions) ** 2) / len(x)
        ),
        "--o",
    )
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title("Frequency of target label versus MSE", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    return
