"""
This file consist the CreateJuniorData class which converts the tutor data into readable data for the
junior agent.
"""
import logging
from pathlib import Path
from typing import List, Union, Tuple, Optional

import defopt
import numpy as np


def prepare_dataset(traindata_path: Path, target_path: Path, dataset_name: str,
                    filtered_obs: Optional[Union[bool, List]] = True,
                    extend: bool = False, seed: Optional[int] = 42):
    """ Prepare/process the training data given by the tutor. The data is seperated into
    training, validation and test dataset, saved at the target_path.

    Args:
        traindata_path: The path to the generated records_* files by the tutor to use.
        target_path: The path to save the dataset files to.
        filtered_obs: Variable, whether or not the obs.to_vect() should be filtered. If False, the original
            observations are used. If a list is provided, the observations are filtered based on the
            ids.
        dataset_name: Name of the dataset which gets used to create the files.
        extend: Whether to extend the exisitng training data instead of creating a new dataset.
        seed: The random seed for shuffling the samples.

    Returns:

    """
    np.random.seed(seed)

    if filtered_obs:
        if isinstance(filtered_obs, list):
            chosen = filtered_obs
        else:
            chosen = [0] + list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
            #       label      timestamp         generator-PQV            load-PQV                      line-PQUI
            chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
            #               line-rho               line switch         line-overload steps          bus switch
            chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(
                range(1164, 1223))
            #          line-cool down steps   substation-cool down steps     next maintenance         maintenance duration
    else:
        chosen = False

    assert traindata_path.exists() and traindata_path.is_dir(), f'{traindata_path} is a valid directory'

    # Change this more globaly #TODO: Check if valid
    npy_files = list(traindata_path.glob('*.npy'))
    assert len(npy_files), f'There are files at {traindata_path}'

    if not extend:
        logging.info(f'Creating final dataset at directory {target_path}')
        make_dataset(target_path=target_path, dataset_name=dataset_name, npy_files=npy_files,
                     chosen=chosen)
        logging.info('Done')
    else:
        train_file_path = target_path / f'{dataset_name}_train.npz'
        extend_dataset(npy_files=npy_files, train_file_path=train_file_path, chosen=chosen)


def merge_dataset(npy_file_paths: List[Path]) -> np.ndarray:
    """Merge the given numpy files at npy_file_paths and return it.

    Note, if the input consists of tuple action IDs, then the IDs are merged to one id!

    Args:
        npy_file_paths:  Input numpy files, created by the Tutor.

    Returns: The unique actions in a merged array.

    """
    assert len(npy_file_paths) > 0, "No files given for merging"

    npy_file_paths = sorted(npy_file_paths)  # This will hopefully make this reproducible

    loaded_actions = [np.load(str(f)) for f in npy_file_paths]
    concated_actions = np.concatenate(loaded_actions, axis=0)
    unique_actions = np.unique(concated_actions, axis=0)

    logging.info(f"All Experience: {concated_actions.shape}")
    logging.info(f"Unique Experience: {unique_actions.shape}")
    logging.info(f"Unique Actions: {len(np.unique(concated_actions[:, 0], axis=0))}")
    return unique_actions


def create_dataset(target_path: Union[Path, str], data: np.ndarray, dataset_name: Optional[str] = "Junior",
                   chosen: Optional[Union[bool, list]] = False):
    """Given all samples data, select the right features and labels and split the dataset up.
    Save the dataset in three different files at target_path.

    Args:
        target_path: Path, where to save the training, validation and testing data for the junior
        data: Dataset for the junior model
        dataset_name: Optional name for the data set.
        chosen: Boolian or List for filtering the observations of the original observations

    Returns: Saves  '{dataset_name}_train.npz','{dataset_name}_val.npz','{dataset_name}_test.npz' in the
    target path

    """
    # Shuffle actions
    np.random.shuffle(data)

    assert target_path.exists() and target_path.is_dir(), "Given target_path is a directory that exists"

    # Subset
    if chosen:
        data = data[:, chosen]

    # Dataset partition
    num_sampling = data.shape[0]
    train_size = num_sampling * 8 // 10
    # validate_size = num_sampling // 10
    test_size = num_sampling // 10
    # s for state, feature; a for action, label.

    # Check whether the input is a Tuple ID or not. If yes
    s_train, a_train = data[:train_size, 1:], data[:train_size, :1].astype(np.int32)
    s_validate, a_validate = data[train_size:-test_size, 1:], data[train_size:-test_size, :1].astype(np.int32)
    s_test, a_test = data[-test_size:, 1:], data[-test_size:, :1].astype(np.int32)

    logging.info(f'TrainSet Size: {len(s_train)}')
    logging.info(f'ValidationSet Size: {len(s_validate)}')
    logging.info(f'TestSet Size: {len(s_test)}')

    train_path = target_path / f'{dataset_name}_train.npz'
    np.savez_compressed(train_path, s_train=s_train, a_train=a_train)

    validation_path = target_path / f'{dataset_name}_val.npz'
    np.savez_compressed(validation_path, s_validate=s_validate, a_validate=a_validate)

    test_path = target_path / f'{dataset_name}_test.npz'
    np.savez_compressed(test_path, s_test=s_test, a_test=a_test)


def load_dataset(dataset_path: Union[str, Path], dataset_name: str) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load the dataset for the training of the Junior model

    Args:
        dataset_path: Path, where to find the dataset
        dataset_name: name of the dataset. There should be '{dataset_name}_train.npz',
        '{dataset_name}_val.npz' and a '{dataset_name}_test.npz' in the target path


    Returns: Returns X,y values for train, val and test data

    """
    train_path = dataset_path / f'{dataset_name}_train.npz'
    train_data = np.load(train_path)

    validation_path = dataset_path / f'{dataset_name}_val.npz'
    validation_data = np.load(validation_path)

    test_path = dataset_path / f'{dataset_name}_test.npz'
    test_data = np.load(test_path)

    return train_data['s_train'], train_data['a_train'], validation_data['s_validate'], \
           validation_data['a_validate'], test_data['s_test'], test_data['a_test']


def make_dataset(target_path: Path, dataset_name: str, npy_files: List[Path],
                 chosen: Optional[Union[bool, list]] = False
                 ):
    """ Join all experience into one file and then execute the create_dataset

    Args:
        target_path: Path, where to write the new experience
        dataset_name: Name of the files
        npy_files: list of np.ndarrays
        chosen: Boolian or List for filtering the observations of the original observations

    Returns: None

    """
    logging.info('Merging data...')
    experience = merge_dataset(npy_files)

    logging.info('Splitting data...')
    create_dataset(target_path=target_path, dataset_name=dataset_name, data=experience,
                   chosen=chosen)


def extend_dataset(npy_files: List[Path], train_file_path: Path, chosen: Optional[Union[bool, list]]) -> object:
    """ Extend existing Training Dataset with new files

    Args:
        npy_files: list of np.ndarrays
        train_file_path: Path of the training data
        chosen: Boolian or List for filtering the observations of the original observations

    Returns: Returns an extended dataset

    """
    logging.info(f'Trying to add data to existing dataset {train_file_path}')
    # 2. Add new data to train set

    # TODO: This is no proper caching!!!
    cache_file_path = Path('merged_new_data.npz')
    if cache_file_path.exists():
        new_experience = np.load(str(cache_file_path))['experience']
    else:
        new_experience = merge_dataset(npy_files)
        # np.savez_compressed(cache_file_path, experience=new_experience)

    final_dataset = np.load(str(train_file_path))
    s_train, a_train = final_dataset['s_train'], final_dataset['a_train']

    np.random.shuffle(new_experience)

    if chosen:
        new_experience = new_experience[:, chosen]

    s_train = np.concatenate([s_train, new_experience[:, 1:]])
    a_train = np.concatenate([a_train, new_experience[:, :1].astype(np.int32)])

    new_filename = train_file_path.name.split('.')[0] + '_v2.npz'
    logging.info(f'Saving new dataset to {new_filename}')
    np.savez_compressed(new_filename, s_train=s_train, a_train=a_train)


if __name__ == "__main__":
    defopt.run(prepare_dataset)
