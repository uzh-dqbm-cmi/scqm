import matplotlib.pyplot as plt


def plot_average_days_to_visit(df_exp, target_name="das283bsr_score", plot_config={}):
    fig, ax = plt.subplots(figsize=(6, 4))

    average_days_to_visit = (
        df_exp.groupby("med_ids")
        .first()
        .reset_index()
        .groupby("drugs")["time_since_med"]
        .mean()
    )
    ax.bar(
        range(len(average_days_to_visit)),
        height=average_days_to_visit.values,
        color=plot_config[target_name]["color"],
    )
    plt.xticks(
        range(len(average_days_to_visit)),
        average_days_to_visit.index.values,
        rotation=90,
    )
    plt.ylabel("Average days")
    plt.title("Average number of days until first visit after medication start")
    ax.set_axisbelow(True)
    ax.grid()
    plt.show()
    return
