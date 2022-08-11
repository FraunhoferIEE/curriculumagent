"""
In this file, the code for the hyper-parameter search of the junior model is provided.
It is optimized to run the NNI framework @https://nni.readthedocs.io/en/stable/
In order to run this code via NNI, you have to start it via the terminal.
NNI then automatically runs through the parameters and selects the best option.
"""
import logging

import ray
import pickle
import numpy as np
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional
import tensorflow as tf

from curriculumagent.junior.hyper_parameter_search.advanced_junior_student import AdvancedJunior
from curriculumagent.junior.junior_student import load_dataset

import nni


def get_default():
    """ Method to get the range of hyper-parameters

    Returns: Default dictionary of hyper-parameters

    """
    hyper_config = {
        "epochs": 10,  # set to 1000, should be adjusted by the A. hyperband
        "learning_rate": 1e-5,
        "activation": "relu",  # TODO:
        "layer1": 1000,
        "layer2": 1000,
        "layer3": 1000,
        "layer4": 1000,
        "dropout1": 0.25,
        "dropout2": 0.25,
        "patience": 0,
        "initializer": "O",
        'batchsize':256
    }
    return hyper_config


def run_hyperopt_training(config, path_to_files: Optional[Path] = None, run_nni: bool = True):
    """ Method for the hyperoptimization training with tune. This method will be executed
    with the config file. The config is specified by tune
    Args:
        config: config file with all the necessary parameters for the Juniort model.
        run_nni: Boolian indicator, whether or not to run the code with NNI. Tipp: Test this code first
        without NNI.

    Returns: The accuracy of the validation run

    """
    if config["initializer"] == "O":
        config["initializer"] = tf.keras.initializers.Orthogonal()
    elif config["initializer"] == "RN":
        config["initializer"] = tf.keras.initializers.RandomNormal()
    elif config["initializer"] == "RU":
        config["initializer"] = tf.keras.initializers.RandomUniform()
    else:
        config["initializer"] = tf.keras.initializers.Zeros()

    config["patience"] = 0

    #
    if path_to_files is None:
        path_to_files = Path('/share/data1/GYM/junior/train')
    s_train, a_train, s_validate, a_validate, _, _ = load_dataset(path_to_files,
                                                                  dataset_name="junior")

    # Standardize:
    scaler = StandardScaler()
    s_tr_t = scaler.fit_transform(s_train)
    s_val_t = scaler.transform(s_validate)

    # First initialize the advanced junior:
    # Note: We do not set a seed, to have more variability!!
    junior = AdvancedJunior(config=config,
                            trainset_size=len(a_train),
                            num_actions=np.max(a_train)+1,
                            run_nni=run_nni)

    history = junior.train(log_dir=None,
                           ckpt_dir=None,
                           patience=None,
                           x_train=s_tr_t, y_train=a_train,
                           x_validate=s_val_t, y_validate=a_validate)
    acc = max(history.history["val_accuracy"])

    if run_nni:
        nni.report_final_result(acc)
    # we return the best validation result:
    return acc


if __name__ == '__main__':
    """
    In order to run this code via NNI, you have to start it via the terminal. 
    
    NNI then automatically runs throught the parameters and selects the best option.     
    See https://nni.readthedocs.io/en/stable/ for the information 
    
    If you want to test the code above set run_nni=False    
    """


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

        run_hyperopt_training(config=params,
                              run_nni=run_nni)

    except Exception as exception:
        logging.exception(exception)
        raise
