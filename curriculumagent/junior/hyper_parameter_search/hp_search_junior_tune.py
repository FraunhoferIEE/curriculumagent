"""
In this file, the code for the hyper-parameter search of the junior model is provided.
This search file however is created to work with tune. In the file we first define the range of the
parameters and afterwards run the tune experiment. See @https://docs.ray.io/en/latest/tune/index.html
for more information.
"""
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

import ray.tune as tune
from ray.tune.suggest.hebo import HEBOSearch


def get_hyperparameter_range():
    """ Method to get the range of hyper-parameters

    Returns: Dictionary with tune distributions

    """
    hyper_config = {
        "epochs": 1000,  # set to 1000, should be adjusted by the A. hyperband
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "activation": tune.choice(["relu", "leaky_relu"]),  # TODO:
        "layer1": tune.uniform(50, 1000),
        "layer2": tune.uniform(0, 1000),
        "layer3": tune.uniform(0, 1000),
        "layer4": tune.uniform(0, 1000),
        "dropout1": tune.uniform(0, 0.5),
        "dropout2": tune.uniform(0, 0.5),
        "patience": tune.choice([0, 5, 10, 15, 20, 25, 30]),
        "initializer": tune.choice([tf.keras.initializers.Orthogonal(),
                                    tf.keras.initializers.RandomNormal(),
                                    tf.keras.initializers.RandomUniform(),
                                    tf.keras.initializers.Zeros()
                                    ])
    }


def run_hyperopt_training(config, path_to_files: Optional[Path] = None):
    """ Method for the hyperoptimization training with tune. This method will be executed
    with the config file. The config is specified by tune
    Args:
        config: config file with all the necessary parameters for the Juniort model.
        path_to_files: Optional path, where the files are stored

    Returns: Dictionary containing the Keras History output

    """
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
                            num_actions=np.max(a_train))

    history = junior.train(log_dir=None,
                           ckpt_dir=None,
                           patience=config["patience"],
                           x_train=s_tr_t, y_train=a_train,
                           x_validate=s_val_t, y_validate=a_validate, )

    # we return the best validation result:
    return history.history

if __name__=="__main__":
    ray.is_initialized()
    ray.init(object_store_memory=32000000000, log_to_driver=False, num_cpus=64,num_gpus=2,dashboard_port=8042) #
    
    config = get_hyperparameter_range()
    print(config)
    scheduler = AsyncHyperBandScheduler(
        time_attr='epochs',
        metric='val_accuracy',
        mode="max",
        max_t=1000
    )

    local_dir="'/share/data1/GYM/junior/hyperopt",
    hebo = HEBOSearch(metric='val_accuracy', mode="max")
    analysis = tune.run(run_hyperopt_training, config=config, search_alg=hebo)
    print("Best hyper-parameters found were: ", analysis.best_config)
    with open(r"analysis.pkl", "wb") as output_file: 
        pickle.dump(analysis, output_file)