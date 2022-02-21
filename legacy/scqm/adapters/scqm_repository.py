"""
Local SCQM data repository

Implements the readout mechanism for the SCQM data (after being preprocessed).
SCQM stands for Swiss Clinical Quality Management. The SCQM Foundation operates a medical registry for inflammatory rheumatic diseases.
"""

import copy
import sys
from pathlib import Path
from functools import partial
from typing import Dict,List, Union, Tuple
import numpy as np

import pandas as pd
from datetime import datetime

from curis.entities import Patient, Visit, Timeseries, PatientQueueItem, Diagnosis
from curis.entities import Drug, DrugsHistory, Measurement, MeasurementsHistory, Timestamp
from curis.ports import PatientRepository

from curis.utils.validation import require_of_types
from curis.utils.format import format_input_to_type
from curis.utils.filesystem import FilesystemError, require_file_exists, require_file_type
from curis.utils.database import get_new_id
from curis.utils.logger import log
#from curis.utils.parallel import parallelize


class SCQMPatientRepository(PatientRepository):
    __slots__ = ["__data_dir", "__tables", "__identifiers"]

    def __init__(self, config: dict = None):

        super().__init__(config)
        require_of_types(self.config['data_dir'], Union[str, Path])

        self.__data_dir = format_input_to_type(self.config['data_dir'], Path)
        self.__patients = None
        self.__tables: Dict[str, pd.DataFrame] = {}
        self.__table_ids = ['v', 'm', 'h', 'eq', 'ra', 'sf', 'se']
        try:
            for table_id in self.__table_ids:
                file_path = (self.__data_dir / (table_id+'_table.csv')).resolve()
                require_file_exists(file_path, f"{file_path} does not exist.")
        except FilesystemError as err:
            log.debug(f"SCQMPatientRepository init failed: {err.message}")
            sys.exit(-1)

    def patients(self, selection: List[PatientQueueItem] = None) -> List[Patient]:
        """Return list of patients according to the selection criteria"""

        if self.__patients is None:
            self.__patients = self.__load_all_patients()
        patients = copy.deepcopy(self.__patients)

        if selection is not None:    # e.o R.A. patients
            # Filter the patients
            valid_patient_ids = [item.patient_id for item in selection]

        return patients

    def __load_all_patients(self) -> List[Patient]:
        """Load and return all patients as a list"""

        log.info(f"Loading of all patients is started...")

        # Loading all tables from .csv files
        for table_id in self.__table_ids:
            self.__tables[table_id] = pd.read_csv(self.__data_dir / (table_id+'_table.csv'))

        # patient_cases = pd.read_csv(self.__data_dir / files[1], sep=',', encoding='utf-8',
        #                             dtype={key: 'category' for key in patient_cases_categorical},
        #                             parse_dates=['encounter_start_date', 'encounter_end_date',
        #                                          'admission_icu_datetime', 'discharge_icu_datetime'])

        table_visit = self.__tables['v']
        table_medication = self.__tables['m']
        patients_ids = table_visit['v.patient_id'].unique()

        patients = list()
        for patient_id in patients_ids:
            # Assign unique idea originated from database
            patient = Patient(patient_id)

            # Demograhic data
            patient_data = table_visit[table_visit['v.patient_id'] == patient_id]
            patient.date_of_birth.set(datetime.strptime(patient_data['v.date_of_birth'].iloc[0], '%Y-%m'))
            patient.gender = patient_data['v.gender'].iloc[0]

            # Artificially creating a visit
            visit = Visit(patient_id)

            visit.add_timeseries(self.__load_timeseries(patient_data))

            # Import Medications
            medication_data = table_medication[table_medication['m.patient_id'] == patient_id]
            patient.add_history(self.__import_drug_exposure(medication_data))

            # Import Measurements
            patient.add_history(self.__import_measurements(patient_data))

            # Assembly patient data
            patient.add_visits(visit)

            patients.append(patient)

        return patients

    def __load_timeseries(self, patient_data: pd.DataFrame) -> Timeseries:
        """Load all visits data to create a timeseries (corresponding) patient"""

        timeseries_id = get_new_id()
        df = patient_data.copy(deep=True)
        drop_columns = ['v.patient_id', 'v.gender', 'v.date_of_birth']
        drop_columns = [col for col in drop_columns if col in df.columns]
        df.drop(columns=drop_columns, inplace=True)
        df.rename(columns={'v.visit_date': 'timestamp'}, inplace=True)
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # Datetime is an arbitrary chosen name for the date information
        df.set_index('timestamp', drop=True, inplace=True, verify_integrity=True)
        df.sort_index(axis=0, inplace=True)

        return Timeseries(df, timeseries_id)

    def __import_drug_exposure(self, drug_data: pd.DataFrame) -> DrugsHistory:
        """Imports patient medication history from the original medication table"""

        drugs = list()
        df = drug_data.copy(deep=True)
        drop_columns = ['m.medication_frequency']
        drop_columns = [col for col in drop_columns if col in df.columns]
        df.drop(columns=drop_columns, inplace=True)

        for index, row in df.iterrows():
            drug = Drug(row['m.medication_id'],  # medication id
                        Timestamp(datetime.strptime(row['m.medication_start_date'], "%Y-%m-%d")
                                  if isinstance(row['m.medication_start_date'], str) else   # medication start date
                                  None),
                        Timestamp(datetime.strptime(row['m.medication_end_date'], "%Y-%m-%d")
                                  if isinstance(row['m.medication_end_date'], str) else     # medication end date
                                  None),
                        'not defined',                              # stop reason
                        row['m.medication_dose'],                   # quantity
                        0,                                          # visit id
                        row['m.medication_drug'],                   # drug_source_value <- name of the drug
                        row['m.medication_route'],                  # name of the drug
                        'not defined',                              # dose_unit_source_value:
                        row['m.medication_generic_drug'],           # drug generic name
                        row['m.medication_drug_classification']     # drug type (classification)
                        )
            drugs.append(drug)
        return drugs

    def __import_measurements(self, measurement_data: pd.DataFrame) -> MeasurementsHistory:
        """Imports all measurement from the original visit table"""

        measurements = list()
        df = measurement_data.copy(deep=True)
        drop_columns = ['v.patient_id', 'v.gender', 'v.date_of_birth']
        drop_columns = [col for col in drop_columns if col in df.columns]
        df.drop(columns=drop_columns, inplace=True)
        df.rename(columns={'v.visit_date': 'timestamp'}, inplace=True)
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # Datetime is an arbitrary chosen name for the date information
        df.set_index('timestamp', drop=True, inplace=True, verify_integrity=True)
        df.sort_index(axis=0, inplace=True)

        for col in df.columns:
            measurements.append(Timeseries(df[col].to_frame(), col))

        return MeasurementsHistory(measurements)

        # for row_idx, row in df.iterrows():
        #     # TODO: management of if the row_idx is not str or int
        #     timestamp = datetime.strptime(row[0], '%Y-%m-%d')
        #     visit_id = int(row[-1])
        #     row = row[:-1]  # Drop visit_id from the row
        #     for m_idx, value in enumerate(row):
        #         # TODO: fix if statement
        #         if m_idx > 0:
        #             m = Measurement(0,                          # id - not in use
        #                             timestamp,                  # time of the measurement
        #                             0.0,                        # values_as_number - not in use
        #                             visit_id,                   # id of the corresponding visit
        #                             row.index[m_idx][2:],     # name of the measured parameter
        #                             None,                       # unit_source_value - not in use
        #                             value                       # actual value (either str, int or float)
        #                             )
        #             measurements.append(m)
        #
        # return measurements

    def save(self, patients: List[Patient], prefix: str = None) -> None:
        """
        Save the the passed in patients list in this repository format
        """

        patients_dfs = parallelize(save_patient, patients)
        df = pd.concat(patients_dfs, ignore_index=True)

        # TODO: Add possibility of saving under another name / path
        filename = f"{prefix if prefix is not None else get_new_id()}_SPO2.csv"
        filepath = Path(self.__data_dir) / filename

        log.info(f"Saving patient list to {filepath.resolve()}")
        df.to_csv(filepath, sep=',', encoding='utf-8', index=False)

        return None


def save_patient(patient: Patient) -> pd.DataFrame:
    """Format a patient information into a single dataframe"""

    timeseries_dfs = list()

    patient_id = patient.get_id()
    for visit in patient.get_visits_by_ids():
        visit_id = visit.get_id()

        for timeseries in visit.get_timeseries_by_ids():
            timeseries_id = timeseries.get_id()

            df = timeseries.get_raw_data().copy(deep=True)
            df.insert(0, 'v.visit_date', df.index)
            df.insert(0, 'timeseries_id', timeseries_id)
            df.insert(0, 'patient_id', patient_id)

            timeseries_dfs.append(df)

    patient_df = pd.concat(timeseries_dfs, ignore_index=True)

    return patient_df
