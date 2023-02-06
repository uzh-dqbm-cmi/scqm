from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
import matplotlib.pyplot as plt


def target_distribution_plot(dataset):
    df_dict = load_dfs_all_data()
    table_patients = df_dict["patients"]
    types = [
        "diagnose_rheumatoid_arthritis",
        "diagnose_ankylosing_spondylitis",
        "diagnose_psoriasis_arthritis",
        "diagnose_undifferentiated_arthritis",
    ]
    color_dict = {
        "Rheumatoid arthritis": "#6CAE75",
        "Ankylosing spondylitis": "#4D7298",
        "Psoriasis arthritis": "#F87575",
        "Undifferentiated arthritis": "#F6BD60",
    }
    patients_per_type = list()
    disease_count = {disease: 0 for disease in types}
    disease_tracker = {disease: [] for disease in types}
    for tp in types:
        patients_per_type.append(
            table_patients[table_patients[tp] == "yes"]["patient_id"].unique()
        )
    for patient in dataset.patient_ids:
        for index, elem in enumerate(types):
            if patient in patients_per_type[index]:
                disease_count[elem] += 1
                disease_tracker[elem].append(patient)

    for df, score, title in zip(
        (dataset.targets_das28_df, dataset.targets_asdas_df),
        ("das283bsr_score", "asdas_score"),
        ("DAS28 distribution", "ASDAS distribution"),
    ):
        plt.figure(figsize=(10, 10))
        if score == "das283bsr_score":
            order = [
                "diagnose_rheumatoid_arthritis",
                "diagnose_ankylosing_spondylitis",
                "diagnose_psoriasis_arthritis",
                "diagnose_undifferentiated_arthritis",
            ]
            label = [
                "Rheumatoid arthritis",
                "Ankylosing spondylitis",
                "Psoriasis arthritis",
                "Undifferentiated arthritis",
            ]
            x = [
                df[df.patient_id.isin(disease_tracker[disease])].das283bsr_score.values
                for disease in order
            ]
        else:
            order = [
                "diagnose_ankylosing_spondylitis",
                "diagnose_rheumatoid_arthritis",
                "diagnose_psoriasis_arthritis",
                "diagnose_undifferentiated_arthritis",
            ]
            label = [
                "Ankylosing spondylitis",
                "Rheumatoid arthritis",
                "Psoriasis arthritis",
                "Undifferentiated arthritis",
            ]
            x = [
                df[df.patient_id.isin(disease_tracker[disease])].asdas_score.values
                for disease in order
            ]
        color = [color_dict[c] for c in label]
        (freq, bins, _) = plt.hist(
            x=x, bins=30, stacked=True, color=color, rwidth=0.85, label=label
        )

        plt.legend()
        plt.xlabel("Score value", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(title, fontsize=14)
        plt.show()
    return
