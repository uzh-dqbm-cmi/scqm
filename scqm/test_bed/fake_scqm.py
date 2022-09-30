import string
import pandas as pd
import string
import random
import time
import numpy as np


def get_patient_ids(num_data: int) -> list:
    """Generate random list of patient ids

    Args:
        num_data (int): number of ids to generate

    Returns:
        list: generated patient ids
    """
    patient_ids = []
    for p in range(num_data):
        patient_ids.extend(
            ["".join(random.choice(string.ascii_lowercase) for i in range(6))]
        )
    return patient_ids


def str_time_prop(
    start: str, end: str, time_format: str = "%m/%d/%Y", prop: float = 0.5
):
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


def random_date(start: str = "01/01/1920", end: str = "01/01/2000", prop=None):
    """
    return date at location prop (decimal between 0 and 1) between dates start and end
    """
    if prop is None:
        prop = random.uniform(0, 1)
    return str_time_prop(start, end, "%m/%d/%Y", prop)


def get_patient_df(num_patients: int = 10) -> pd.DataFrame:
    """create random df of patient data

    Args:
        num_patients (int, optional): Number of patients. Defaults to 10.

    Returns:
        pd.DataFrame: generated df
    """
    patient_ids = get_patient_ids(num_patients)
    gender = random.choices(["male", "female"], k=num_patients)
    date_of_birth = [random_date() for p in range(num_patients)]
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "gender": gender,
            "date_of_birth": date_of_birth,
            "date_first_symptoms": np.nan,
            "date_diagnosis": np.nan,
        }
    )


def get_haq_df(patient_df: pd.DataFrame, visit_df: pd.DataFrame) -> pd.DataFrame:
    """Generate random haq df from patients in patient df

    Args:
        patient_df (pd.DataFrame): patient df

    Returns:
        pd.DataFrame: haq df
    """
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 2)
    haq_values = random.choices(range(10), k=len(ids))
    # select random subset of visit_df corresponding to patients in ids
    tmp = visit_df[visit_df.patient_id.isin(ids)].sample(n=len(ids))
    ids = tmp.patient_id.values
    uid_num = tmp.uid_num.values
    dates = tmp.date.values
    # uid_num = random.sample(range(4000), k=len(ids))
    # dates = [
    #     random_date(
    #         start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
    #     )
    #     for patient_id in ids
    # ]
    return pd.DataFrame(
        {"patient_id": ids, "date": dates, "uid_num": uid_num, "haq_score": haq_values}
    )


def get_socioeco_df(patient_df: pd.DataFrame, visit_df: pd.DataFrame) -> pd.DataFrame:
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 2)
    smoker_values = random.choices(
        [
            "i_am_currently_smoking",
            "i_have_never_smoked",
            "i_am_a_former_smoker_for_more_than_a_year",
            "never_been_smoking",
            "smoking_currently",
            "a_former_smoker",
        ],
        k=len(ids),
    )
    tmp = visit_df[visit_df.patient_id.isin(ids)].sample(n=len(ids))
    ids = tmp.patient_id.values
    uid_num = tmp.uid_num.values
    dates = tmp.date.values

    return pd.DataFrame(
        {"patient_id": ids, "date": dates, "uid_num": uid_num, "smoker": smoker_values}
    )


def get_basdai_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Generate random basdai df from patients in patient df

    Args:
        patient_df (pd.DataFrame): patient df

    Returns:
        pd.DataFrame: haq df
    """
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 2)
    basdai_values = random.choices(range(10), k=len(ids))
    event_id = random.sample(range(4000), k=len(ids))
    event_id = [str(elem) for elem in event_id]
    dates = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    return pd.DataFrame(
        {
            "patient_id": ids,
            "date": dates,
            "basdai_score": basdai_values,
            "event_id": event_id,
        }
    )


def get_radai_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 2)
    pain_level_values = random.choices(range(10), k=len(ids))
    activity_radai = random.choices(range(10), k=len(ids))
    morning_stiffness = random.choices(
        [
            "no_morning_stiffness",
            "1_to_2_hours",
            "30_minutes_to_1_hour",
            "2_to_4_hours",
            "less_than_30_minutes",
            "all_day",
            "more_than_4_hours",
        ],
        k=len(ids),
    )
    event_id = random.sample(range(4000), k=len(ids))
    event_id = [str(elem) for elem in event_id]
    dates = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    return pd.DataFrame(
        {
            "patient_id": ids,
            "date": dates,
            "pain_level_today_radai": pain_level_values,
            "morning_stiffness_duration_radai": morning_stiffness,
            "activity_of_rheumatic_disease_today_radai": activity_radai,
            "uid_num": event_id,
        }
    )


def get_visit_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    ids = random.choices(patient_df["patient_id"].values, k=len(patient_df) * 6)
    dates = [
        random_date(
            start=patient_df[patient_df.patient_id == patient_id].date_of_birth.item()
        )
        for patient_id in ids
    ]
    uid_num = random.sample(range(100000), k=len(ids))
    uid_num = [str(elem) for elem in uid_num]
    das283_bsr = random.choices(range(6), k=len(ids))
    asdas_score = random.choices(range(6), k=len(ids))
    das283_bsr = drop_values(das283_bsr, nan_prop=0.5)
    asdas_score = drop_values(asdas_score, nan_prop=0.7)
    weight_kg = random.choices(range(40, 200), k=len(ids))
    return pd.DataFrame(
        {
            "patient_id": ids,
            "uid_num": uid_num,
            "date": dates,
            "das283bsr_score": das283_bsr,
            "asdas_score": asdas_score,
            "weight_kg": weight_kg,
            "n_swollen_joints": np.nan,
            "n_painfull_joints": np.nan,
            "bsr": np.nan,
            "n_painfull_joints_28": np.nan,
            "anti_ccp": np.nan,
            "ra_crit_rheumatoid_factor": np.nan,
            "height_cm": np.nan,
            "crp": np.nan,
            "hb": np.nan,
            "n_enthesides": np.nan,
            "mda_score": np.nan,
            "joints_type": np.nan,
        }
    )


def drop_values(list_: list, nan_prop: float) -> list:
    """randomly replace some elements of the list with nans

    Args:
        list_ (list): initial list of values
        nan_prop (np.float): proportion of element to be replaced
    """
    list_with_nan = list_.copy()
    indices_to_drop = np.random.choice(
        range(len(list_)), size=int(nan_prop * len(list_)), replace=False
    )
    for index in indices_to_drop:
        list_with_nan[index] = np.nan
    return list_with_nan


def get_med_df(patient_df: pd.DataFrame) -> pd.DataFrame:
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
    med_id = random.sample(range(20000), k=len(ids))
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


def get_df_dict(num_patients: int = 10) -> dict:
    patient_df = get_patient_df(num_patients)
    visit_df = get_visit_df(patient_df)
    haq_df = get_haq_df(patient_df, visit_df)
    socioeco_df = get_socioeco_df(patient_df, visit_df)
    med_df = get_med_df(patient_df)
    basdai_df = get_basdai_df(patient_df)
    radai_df = get_radai_df(patient_df)
    df_dict = {
        "patients": patient_df,
        "visits": visit_df,
        "haq": haq_df,
        "socioeco": socioeco_df,
        "medications": med_df,
        "basdai": basdai_df,
        "radai5": radai_df,
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
    columns_basdai = ["patient_id", "date", "basdai_score"]
    patient_df = get_patient_df(100)
    visit_df = get_visit_df(patient_df)
    haq_df = get_haq_df(patient_df, visit_df)
    socioeco_df = get_socioeco_df(patient_df, visit_df)
    med_df = get_med_df(patient_df)
    basdai_df = get_basdai_df(patient_df)

    save = True
    if save:
        patient_df.to_csv("scqm/test_bed/dummy_data/patients.csv", index=False)
        haq_df.to_csv("scqm/test_bed/dummy_data/haq.csv", index=False)
        visit_df.to_csv("scqm/test_bed/dummy_data/visits.csv", index=False)
        med_df.to_csv("scqm/test_bed/dummy_data/medications.csv", index=False)
        basdai_df.to_csv("scqm/test_bed/dummy_data/basdai.csv", index=False)
    print("End of file")
