import string
import pandas as pd
import string
import random
import time
import numpy as np


def get_patient_ids(num_data):
    patient_ids = []
    for p in range(num_data):
        patient_ids.extend(
            ["".join(random.choice(string.ascii_lowercase) for i in range(5))]
        )
    return patient_ids


def str_time_prop(start, end, time_format, prop):
    """
    https://stackoverflow.com/questions/553303/generate-a-random-date-between-two-other-dates

    Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start="01/01/1920", end="01/01/2000", prop=None):
    """
    return date at location prop (decimal between 0 and 1) between dates start and end
    """
    if prop is None:
        prop = random.uniform(0, 1)
    return str_time_prop(start, end, "%m/%d/%Y", prop)


def get_patient_df(num_patients=10):
    patient_ids = get_patient_ids(num_patients)
    gender = random.choices(["male", "female"], k=num_patients)
    date_of_birth = [random_date() for p in range(num_patients)]
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "gender": gender,
            "date_of_birth": date_of_birth,
            "anti_ccp": np.nan,
            "ra_crit_rheumatoid_factor": np.nan,
            "date_first_symptoms": np.nan,
            "date_diagnosis": np.nan,
        }
    )


def get_haq_df(patient_df):
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 2)
    haq_values = random.choices(range(10), k=len(ids))
    uid_num = random.sample(range(4000), k=len(ids))
    dates = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    return pd.DataFrame(
        {"patient_id": ids, "date": dates, "uid_num": uid_num, "haq_score": haq_values}
    )


def get_visit_df(patient_df):
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 6)
    dates = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    uid_num = random.sample(range(4000), k=len(ids))
    uid_num = [str(elem) for elem in uid_num]
    das283_bsr = random.choices(range(6), k=len(ids))
    weight_kg = random.choices(range(40, 200), k=len(ids))
    return pd.DataFrame(
        {
            "patient_id": ids,
            "uid_num": uid_num,
            "date": dates,
            "das283bsr_score": das283_bsr,
            "weight_kg": weight_kg,
            "n_swollen_joints": np.nan,
            "n_painfull_joints": np.nan,
            "bsr": np.nan,
            "n_painfull_joints_28": np.nan,
            "height_cm": np.nan,
            "crp": np.nan,
        }
    )


def get_med_df(patient_df):
    mapping = {
        "methotrexate": "cSDMARD",
        "etanercept": "bDMARD",
        "sulfasalazine": "cSDMARD",
        "prednisone": "steroid",
        "tofacitinib": "tsDMARD",
    }
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 6)
    dates_start = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    dates_stop = [random_date(start=date_start) for date_start in dates_start]
    med_id = random.sample(range(4000), k=len(ids))
    med_id = ["med_" + str(elem) for elem in med_id]
    drug_name = random.choices(
        ["methotrexate", "prednisone", "sulfasalazine", "etanercept", "tofacitinib"],
        k=len(ids),
    )
    drug_category = [mapping[drug] for drug in drug_name]

    return pd.DataFrame(
        {
            "patient_id": ids,
            "med_id": med_id,
            "medication_start_date": dates_start,
            "medication_end_date": dates_stop,
            "medication_generic_drug": drug_name,
            "medication_drug_classification": drug_category,
            "medication_dose": np.nan,
        }
    )


def get_df_dict(num_patients=10):
    patient_df = get_patient_df(num_patients)
    haq_df = get_haq_df(patient_df)
    visit_df = get_visit_df(patient_df)
    med_df = get_med_df(patient_df)
    df_dict = {
        "patients": patient_df,
        "visits": visit_df,
        "haq": haq_df,
        "medications": med_df,
    }
    return df_dict


if __name__ == "__main__":

    columns_patients = ["patient_id", "gender", "date_of_birth"]
    columns_visits = ["uid_num", "patient_id", "date", "das283_bsr", "other_feature"]
    columns_medications = [
        "patient_id",
        "med_id",
        "medication_start_date",
        "medication_end_date",
        "drug_name",
        "drug_category",
    ]
    columns_haq = ["patient_id", "date", "haq"]

    patient_df = get_patient_df()
    haq_df = get_haq_df(patient_df)
    visit_df = get_visit_df(patient_df)
    med_df = get_med_df(patient_df)

    print("End of file")

# general_df = df_dict["patients"][
#     [
#         "patient_id",
#         "date_of_birth",
#         "gender",
#         "anti_ccp",
#         "ra_crit_rheumatoid_factor",
#         "date_first_symptoms",
#         "date_diagnosis",
#     ]
# ]
# med_df = df_dict["medications"][
#     [
#         "patient_id",
#         "med_id",
#         "medication_generic_drug",
#         "medication_drug_classification",
#         "medication_dose",
#         "medication_start_date",
#         "medication_end_date",
#     ]
# ]
# visits_df = df_dict["visits"][
#     [
#         "patient_id",
#         "uid_num",
#         "date",
#         "weight_kg",
#         "das283bsr_score",
#         "n_swollen_joints",
#         "n_painfull_joints",
#         "bsr",
#         "n_painfull_joints_28",
#         "height_cm",
#         "crp",
#     ]
# ]

# haq_df = df_dict["haq"][["patient_id", "uid_num", "date", "haq_score"]]