"""
Pipeline responsible for features extraction
"""

import sys
import yaml

from curis.domain.transform.visit import SplitTimeseries, MinDurationTimeseries, MaxLengthTimeseries, MinDatapoints
from curis.domain.transform.timeseries import FillMissingValues, MissingValuesTimeInterval
from curis.domain.transform.timeseries import EventDetection, TargetDefinition
from curis.factories import PreprocessingFactory
from curis.utils.logger import log, configure_log_to_file

from scqm.adapters.scqmpatientrepository import SCQMPatientRepository


def preprocess_features(config_file: str):

    # Run ID
    run_id = 'chopped'
    print(f"Sync check: 4")

    # Configuration
    with open(config_file) as file:
        configs = yaml.full_load(file)

    log_dir = configs['logging']['dir']
    configure_log_to_file('exec', f"{log_dir}/{run_id}.log")
    log.info(f"Preprocessing data - Run ID: {run_id}")

    # Initialize the adapters to pass to the use case factory
    patient_repository = SCQMPatientRepository(configs['patient_repository'])
    split_timeseries = SplitTimeseries(config=configs['transform']['visit']['split'])
    min_duration_timeseries = MinDurationTimeseries(config=configs['transform']['visit']['min_duration'])
    max_length_timeseries = MaxLengthTimeseries(config=configs['transform']['visit']['max_length'])
    min_datapoints = MinDatapoints(config=configs['transform']['visit']['min_datapoints'])

    # TODO: Add timeseries transform to check range of values
    hypoxia_detection = EventDetection(config=configs['transform']['timeseries']['hypoxia_detection'])
    ventilator_detection = EventDetection(config=configs['transform']['timeseries']['ventilator_detection'])
    fill_missing = FillMissingValues(config=configs['transform']['timeseries']['fill_missing_values'])
    time_interval = MissingValuesTimeInterval(config=configs['transform']['timeseries']['missing_values_time_interval'])
    target_definition = TargetDefinition(config=configs['transform']['timeseries']['target_definition'])

    # ---------------------------------- PREPROCESSING ------------------------------------- #
    factory = PreprocessingFactory(patient_repository=patient_repository,
                                   patient_transforms=[split_timeseries, min_duration_timeseries,
                                                       max_length_timeseries, min_datapoints,
                                                       hypoxia_detection, ventilator_detection,
                                                       fill_missing, time_interval, target_definition],
                                   run_id=run_id
                                   )

    # Run the pipeline
    factory.load_patient_list().invoke()
    factory.apply_patient_transforms().invoke()
    factory.save_patient_list().invoke()

    # Save the configuration file
    with open(f"{log_dir}/{run_id}_configs.yaml", 'w') as file:
        yaml.dump(configs, file, indent=4)


if __name__ == "__main__":
    preprocess_features(sys.argv[1])