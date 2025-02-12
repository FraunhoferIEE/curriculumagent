"""In this file, the code for the hyperparameter search of the junior model is provided.
It is optimized to run the NNI framework @https://nni.readthedocs.io/en/stable/
In order to run this code via NNI, you have to start it via the terminal.
NNI then automatically runs through the parameters and selects the best option.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import nni
import numpy as np
from nni import Experiment
from nni.experiment.config import ExperimentConfig
from sklearn.preprocessing import StandardScaler

from curriculumagent.junior.junior_student import Junior
from curriculumagent.junior.junior_student import load_dataset


def get_default():
    """Method to get the range of hyperparameters.

    Args:
        None.

    Returns:
        Default dictionary of hyperparameters.

    """
    hyper_config = {
        "epochs": 10,  # set to 1000, should be adjusted by the A. hyperband
        "learning_rate": 1e-5,
        "activation": "relu",
        "layer1": 1000,
        "layer2": 1000,
        "layer3": 1000,
        "layer4": 1000,
        "dropout1": 0.25,
        "dropout2": 0.25,
        "patience": 0,
        "initializer": "O",
        'batchsize': 256
    }
    return hyper_config


def run_one_experiment(config,action_space_file, path_to_files: Path, run_nni: bool = True):
    """ Method for the hyper-optimization training with tune. This method will be executed
    with the config file. The config is specified by tune.

    Args:
        action_space_file:
        config: Config file with all the necessary parameters for the Junior model.
        run_nni: Boolean indicator, whether to run the code with NNI. Tip: Test this code first without NNI.

    Returns:
        The accuracy of the validation run.

    """

    config["patience"] = 0
    name = [a for a in os.listdir(path_to_files) if "_train.npz" in a][0][:-10]

    logging.info(f"Assuming the name {name} for the data files")
    s_train, a_train, s_validate, a_validate, _, _ = load_dataset(path_to_files, dataset_name=name)

    # Standardize:
    scaler = StandardScaler()
    s_tr_t = scaler.fit_transform(s_train)
    s_val_t = scaler.transform(s_validate)

    # First initialize the advanced junior:
    # Note: We do not set a seed, to have more variability!!
    junior = Junior(action_space_file=action_space_file,config=config, run_nni=run_nni)

    history = junior.train(
        run_name="experiment",
        dataset_path=path_to_files,
        dataset_name=name,
        target_model_path = path_to_files,
    )
    acc = max(history.history["val_accuracy"])

    if run_nni:
        nni.report_final_result(acc)
    # we return the best validation result:
    return acc


def run_nni_experiment(path_to_files: Path,
                       max_trial_number: Optional[int] = 100,
                       experiment_working_directory: Optional[Path] = None):
    """ This is the method to run the Hyper-parameter search with NNI based on https://nni.readthedocs.io/.
    In the method, we initialize a model and then run the script within a python method.

    Args:
        path_to_files: Directory where to find the dataset of the tutor, which is then loaded with load_dataset.
        path_to_config: Optional directory, where to find the config of the experiment. If not supplied, we use the
            directory of this file.
        experiment_working_directory: Where the experiment files should be saved.

    Note:
          This method is still experimental and depends on the machine and your directory. A working alternative
          is the start of the experiment via nnictl in the terminal

    Returns:
        None
    """
    exp_config = ExperimentConfig(experiment_name='Junior',
                                  experiment_type="hpo",
                                  trial_concurrency=4,
                                  max_experiment_duration='48h',
                                  max_trial_number=max_trial_number,
                                  training_service_platform='local',
                                  search_space_file=Path(__file__).parent / 'search_space.json',
                                  use_annotation=False,
                                  advisor={'name': 'BOHB',
                                           'class_args': {'optimize_mode': 'maximize',
                                                          'min_budget': 1,
                                                          'max_budget': 100,
                                                          'eta': 3,
                                                          'min_points_in_model': 17,
                                                          'top_n_percent': 20,
                                                          'num_samples': 128,
                                                          'random_fraction': 0.33,
                                                          'bandwidth_factor': 3.0,
                                                          'min_bandwidth': 0.001}},
                                  trial_command=f'python hyper_parameter_search_junior_nni.py --path_to_files={path_to_files}',
                                  trial_code_directory=Path(__file__).parent,
                                  trial_gpu_number=0)

    assert any(
        [".npz" in p for p in os.listdir(path_to_files)]), "Did not find any .npz file, i.e., the tutor experience!"

    # Init Experiment:
    if experiment_working_directory:
        assert experiment_working_directory.is_dir(), f"The dir {experiment_working_directory} does not exist!"
        exp_config.experiment_working_directory = experiment_working_directory

    experiment = Experiment(exp_config)
    logging.info(f"Run the NNI experiment with the following config: {experiment.config}")
    experiment.run(8080, wait_completion=True)
    experiment.stop()
    logging.info("Reverting working directory")


if __name__ == "__main__":
    """ This main method is needed to run the NNI experiment. 
    Either you start the experiment with the run_nni_experiment method, or via the terminal.

    NNI then automatically runs throught the parameters and selects the best option.
    See https://nni.readthedocs.io/en/stable/ for the information

    If you want to test the code above set run_nni=False
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run NNI Experiment')
    parser.add_argument('--path_to_files', metavar='path', required=True,
                        help='Path, where to find the junior train.npz files. This is required.')
    args = parser.parse_args()

    run_nni = True
    params = get_default()
    try:
        # get parameters form tuner
        if run_nni:
            tuner_params = nni.get_next_parameter()
            logging.info(tuner_params)
            tuner_params['epochs'] = tuner_params['TRIAL_BUDGET'] * 10 + 5
            params.update(tuner_params)
            logging.info(params)

        run_one_experiment(config=params,
                           path_to_files=args.path_to_files,
                           run_nni=run_nni)

    except Exception as exception:
        logging.exception(exception)
        raise
