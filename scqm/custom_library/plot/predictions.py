import matplotlib.pyplot as plt
import random


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
    elif target_name == "basdai_score":
        patients = [
            patient
            for patient in dataset.test_ids
            if dataset[patient].target_name == "basdai_score"
        ]
        ylabel = "basdai"
    if patient_id is None:
        p = random.sample(patients, 1)[0]
    else:
        p = patient_id
    if plot_time:
        targets = df[df.patient_id == p].targets
        predictions = df[df.patient_id == p].predictions
        dates = df[df.patient_id == p].prediction_dates
        plt.figure(figsize=(7, 5))
        plt.xticks(rotation=90)
        plt.plot(dates, targets, "*", label="true values")
        plt.plot(dates, predictions, "*", label="predictions")
        # plt.xticks(range(1, len(targets)+1))
        plt.xlabel("Date")
        plt.ylim((0, 10))
        plt.ylabel(ylabel)
        plt.legend()
    else:
        targets = df[df.patient_id == p].targets
        predictions = df[df.patient_id == p].predictions
        plt.plot(range(1, len(targets) + 1), targets, "*", label="true values")
        plt.plot(range(1, len(targets) + 1), predictions, "*", label="predictions")
        plt.xticks(range(1, len(targets) + 1))
        plt.xlabel("Predicted visit")
        plt.ylim((0, 10))
        plt.ylabel(ylabel)
        plt.legend()

    return df[df.patient_id == p]
