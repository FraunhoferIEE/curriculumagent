"""
This file is the advanced agent constisting of a model that can cover different action spaces, rllib
training, scaling and other additional features.
"""

import logging
import os
import datetime
import pickle
from pathlib import Path
from typing import Optional, Union, List

import grid2op
from sklearn.base import BaseEstimator
import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.keras.models import Model
from grid2op.Agent import BaseAgent
from curriculumagent.common.utilities import find_best_line_to_reconnect, \
    is_legal, array2action, split_action_and_return


class MyAgent(BaseAgent):
    def __init__(self, action_space,
                 model_path: Union[Path, str],
                 this_directory_path: Optional[str] = './',
                 action_space_path: Optional[Union[Path, List[Path]]] = None,
                 filtered_obs: Optional[bool] = True,
                 scaler: Optional[BaseEstimator] = None,
                 best_action_threshold=0.95):
        """ The new advanced agent.

        In contrast to the original agent, this agent enables the implementation of tuple and triple actions,
        as well as use either a keras model or a model from rllib which is a AutoTrackable model.

        Next to the difference in the models and actions, this agent also has the ability to transform the
        observations based on a provided scaler and/or filter them accordingly.

        Args:
            action_space: Action Space of the Grid2Op Enviornment
            model_path: Path where to find the rllib model or Keras model
            this_directory_path: Path of the submission directory
            action_space_path: path, where to find the action sets
            filtered_obs: Boolean, whether to filter the observation
            scaler: Optional Scaler for the neural network
            best_action_threshold: Threshold, when to stop searching for the results.
        """
        # Initialize a new agent.
        BaseAgent.__init__(self, action_space=action_space)

        # Collect action set:
        self.actions = self.__collect_action(this_directory_path=this_directory_path,
                                             action_space_path=action_space_path)

        if filtered_obs:
            if isinstance(filtered_obs, list):
                self.chosen = filtered_obs
            else:
                self.chosen = self.__get_default_index()
        else:
            self.chosen = None

        # Load Model:
        try:
            self.model: Model = tf.keras.models.load_model(model_path)
            self.model.compile()

        except (IndexError,AttributeError):
            self.model: AutoTrackable = tf.saved_model.load(str(model_path))

        self.scaler = scaler
        self.last_step = datetime.datetime.now()
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None
        self.best_action_threshold = best_action_threshold

    def act(self, observation: grid2op.Observation.BaseObservation,
            reward: float, done: bool) -> grid2op.Action.BaseAction:
        """ Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Args:
            observation: Grid2Op Observation
            reward: Reward of the previous action
            done: Whether the agent is done

        Returns: A suitable Grid2Op action

        """

        # Similar to the Tutor, we check whether there is some remaining action, based on previous
        # selected tuples
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                if is_legal(next_action, observation):
                    return next_action
            except StopIteration:
                self.next_actions = None

        tnow = observation.get_time_stamp()
        if self.last_step + datetime.timedelta(minutes=5) != tnow:
            logging.info('\n\nscenario changes！')
            self.recovery_stack = []
        self.last_step = tnow

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure with low threshold
        action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))

        if observation.rho.max() < self.best_action_threshold:  # fixed threshold
            return action

        # Now, case dangerous:
        o, _, d, _ = observation.simulate(action)
        min_rho = o.rho.max() if not d else 10

        logging.info(f'{observation.get_time_stamp()}s, heavy load,'
                     f' line-{observation.rho.argmax()}d load is {observation.rho.max()}')

        idx_chosen = None

        if isinstance(self.model, Model):
            sorted_actions = self.__get_keras_actions(obs=observation)
        else:
            sorted_actions = self.__get_tf_actions(obs=observation)

        for k, idx in enumerate(sorted_actions):
            a = array2action(action_space=self.action_space,
                             action_vect=self.actions[idx, :])
            a = find_best_line_to_reconnect(obs=observation,
                                            original_action=a)
            if not is_legal(a, observation):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() <= self.best_action_threshold:
                logging.info(f'take action {idx}, max-rho to {obs.rho.max()},'
                             f' simulation times: {k + 1}')
                idx_chosen = idx
                break

            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                idx_chosen = idx

        if idx_chosen:
            self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
            next_action = next(self.next_actions)

            return next_action
        else:
            return find_best_line_to_reconnect(obs=observation,
                                               original_action=self.action_space({}))

    def __collect_action(self, this_directory_path: str, action_space_path: Union[Path, List[Path]]) -> np.ndarray:
        """ Check the action space path for the different action set.

        Args:
            this_directory_path: Directory of the submission files
            action_space_path: Optional action space path

        Returns:

        """
        if action_space_path is None:
            actions62 = np.load(os.path.join(this_directory_path, 'actions62.npy'))
            actions146 = np.load(os.path.join(this_directory_path, 'actions146.npy'))
            actions = np.concatenate((actions62, actions146), axis=0)
        else:
            if isinstance(action_space_path, Path):
                if action_space_path.is_file():
                    logging.info(f"Action_space_path {action_space_path} is a file and will be loaded.")
                    actions = np.load(str(action_space_path))
                elif action_space_path.is_dir():
                    logging.info(f"Action_space_path {action_space_path} is a path. All available action files "
                                 f" will be loaded.")
                    all_action_files = [act for act in os.listdir(action_space_path) if
                                        "actions" in act and ".npy" in act]

                    if not all_action_files:
                        raise FileNotFoundError("No actions files were found!")

                    loaded_files = []
                    for act in all_action_files:
                        if "actions" in act and ".npy" in act:
                            loaded_files.append(np.load(action_space_path / act))

                    actions = np.concatenate(loaded_files, axis=0)
            elif isinstance(action_space_path, list):
                logging.info(f"Action_space_path {action_space_path} is a list containing multiple actions.")
                for act_path in action_space_path:
                    if isinstance(act_path, Path):
                        assert act_path.is_file()
                    else:
                        os.path.isfile(act_path)
                loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
                actions = np.concatenate(loaded_files, axis=0)
            else:
                raise ValueError(f"The action_space_path variable {action_space_path} does neither consist of a single "
                                 f"action nor of a path where actions can be found.")
        return actions

    def __get_default_index(self) -> List:
        """ Method to receive the default chosen values to filter the obs.to_vet

        Note: Contrary to the Tutor, we now have one value less. Thus the list starts at 1

        Returns: list of chosen values

        """
        # Assign label & features

        chosen = list(range(1, 6)) + list(range(6, 72)) + list(range(72, 183)) + list(range(183, 655))
        #       label      timestamp         generator-PQV            load-PQV                      line-PQUI
        chosen += list(range(655, 714)) + list(range(714, 773)) + list(range(773, 832)) + list(range(832, 1009))
        #               line-rho               line switch         line-overload steps          bus switch
        chosen += list(range(1009, 1068)) + list(range(1068, 1104)) + list(range(1104, 1163)) + list(range(1163, 1222))
        #          line-cool down steps   substation-cool down steps     next maintenance         maintenance duration
        return chosen

    def __get_keras_actions(self, obs: grid2op.Observation.BaseObservation):
        """ Method to get the keras actions:

        Args:
            obs: Current observations

        Returns: numpy with the sorted actions:

        """
        features = obs.to_vect()

        # Transform if wanted:
        if self.chosen:
            features = features[self.chosen]
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1)).reshape(-1, )

        print(features)

        _, a_pred, _ = self.model.predict(features)
        a_pred = a_pred._numpy()
        sorted_actions = a_pred[0, :].argsort()[::-1]
        return sorted_actions

    def __get_tf_actions(self, obs: grid2op.Observation.BaseObservation):
        """ Method to get the tf actions:

        Args:
            obs: Current observations

        Returns: sorted numpy array with actions

        """
        features = obs.to_vect()

        # Transform if wanted:
        if self.chosen:
            features = features[self.chosen]
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1)).reshape(-1,)

        f = self.model.signatures['serving_default']
        out = f(observations=tf.convert_to_tensor(features.reshape(1, -1)),
                timestep=tf.convert_to_tensor(0, dtype=tf.int64),
                is_training=tf.convert_to_tensor(False))

        # Collect the softmax over all actions
        prob_of_action = tf.nn.softmax(out['action_dist_inputs']).numpy().reshape(-1, )
        sorted_actions = prob_of_action.argsort()[::-1]
        return sorted_actions


def make_agent(env, this_directory_path):
    action_list = [Path(this_directory_path) / "action_sets" / "actionspace_tuples.npy",
                   Path(this_directory_path) / "action_sets" / "actionspace_nminus1.npy",
                   ]
    with open(Path(this_directory_path) / "scaler_junior.pkl", "rb") as fp:  # Pickling
        scaler = pickle.load(fp)

    my_agent = MyAgent(action_space=env.action_space,
                       model_path=Path(this_directory_path) / "rllib_model",
                       this_directory_path=Path(this_directory_path),
                       action_space_path=action_list,
                       filtered_obs=True,
                       scaler=scaler)
    return my_agent

if __name__ == "__main__":
    # Due to commitproblems, the rllib_model had to be zipped. Therefore, we first have to unzip the file:
    import zipfile
    this_dir = os.getcwd()
    with zipfile.ZipFile(Path(this_dir) / "rllib_model.zip", 'r') as zip_ref:
        zip_ref.extractall(this_dir)
    env = grid2op.make("l2rpn_neurips_2020_track1_small")
    agent = make_agent(env,this_dir)
    obs = env.reset()
    act = agent.act(obs,0,False)
