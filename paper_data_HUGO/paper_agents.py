import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple

import grid2op
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator

from curriculumagent.common.obs_converter import obs_to_vect
from curriculumagent.common.utilities import find_best_line_to_reconnect, is_legal, revert_topo, simulate_action, \
    split_action_and_return, set_bus_from_topo_vect
from curriculumagent.submission.my_agent import MyAgent


class TopologyAgent2(MyAgent):
    def __init__(
            self,
            action_space,
            model_path: Union[Path, str],
            action_space_path: Optional[Union[Path, List[Path]]] = None,
            this_directory_path: Optional[str] = "./",
            subset: Optional[bool] = False,
            scaler: Optional[BaseEstimator] = None,
            best_action_threshold: float = 0.95,
            topo: Optional[bool] = False,
            check_overload: Optional[bool] = False,
            max_action_sim: Optional[int] = 3,
            action_space_file: Optional[str] = None,
            topology_actions: [Union[Path, str]] = None,
            topo_threshold=0.85
    ):
        """The topology agent. We provide an numpy array to select the topology

        Note:
            If you just want to pass this agent as submission without the CurriculumAgent, copy the content
            of the common dir into this directory. Further, add the model and actions to complete it.

        Args:
            action_space: Action Space of the Grid2Op Enviornment
            model_path: Path where to find the rllib model or Keras model
            action_space_path: path, where to find the action sets. This is required to run the agent
            this_directory_path: Path of the submission directory
            subset: Boolean, whether to filter the observation
            scaler: Optional Scaler for the neural network
            best_action_threshold: Threshold, when to stop searching for the results.
            topo: Booling indicator, whether the agent should revert to original topology if it is possible
            check_overload: Boolean, whether to simulate a stress of the generation and load
            max_action_sim: Define how many substations should be simulated. Default is 3
            candidate. If you want to select all, it has to be the number of actions. For a more rapid simulation, you
            can just select fewer values.
            action_space_file: Separate Topology file
            topology_actions: File, where to find the topology actions
        """
        # Initialize the MyAgent methode.
        super().__init__(action_space=action_space,
                         model_path=model_path,
                         action_space_path=action_space_path,
                         this_directory_path=this_directory_path,
                         subset=subset,
                         scaler=scaler,
                         best_action_threshold=best_action_threshold,
                         topo=topo,
                         check_overload=check_overload,
                         max_action_sim=max_action_sim,
                         )

        if topology_actions:
            self.topologies = np.load(topology_actions)
        else:
            raise AttributeError

        self.topo_threshold = topo_threshold

    def act_with_id(
            self, observation: grid2op.Observation.BaseObservation) -> Tuple[grid2op.Action.BaseAction, int]:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Note: this method was primarily written to plug it into the generate experience methode

        Args:
            observation: Grid2Op Observation

        Returns: A suitable Grid2Op action

        """

        # Similar to the Tutor, we check whether there is some remaining action, based on previous
        # selected tuples
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        ###################################
        # Default action:
        if self.topo:
            action_array = revert_topo(self.action_space, observation)
            default_action = self.action_space.from_vect(action_array)
            default_action = find_best_line_to_reconnect(obs=observation,
                                                         original_action=default_action)
        else:
            default_action = self.action_space({})
            default_action = find_best_line_to_reconnect(obs=observation,
                                                         original_action=default_action)

        idx_chosen = None
        ###################################
        # Option 0: Secure with Rho below topo_threshold
        if observation.rho.max() < self.topo_threshold:  # Below < 0.85
            return default_action, -1

        ###################################
        # Option 1: Critical impact: Check for topologies  with Rho above topo_threshold
        if observation.rho.max() < self.best_action_threshold:

            min_rho = sim_rho = observation.rho.max()
            topo_act = None
            idx = None
            for i, top_a in enumerate(self.topologies):
                transf_act = set_bus_from_topo_vect(observation.topo_vect, top_a, self.action_space)
                # Simulate:
                simulator = observation.get_simulator()
                simulator_stressed = simulator.predict(act=transf_act)
                if simulator_stressed.converged:
                    sim_rho = simulator_stressed.current_obs.rho.max()
                    if sim_rho < min_rho:
                        topo_act = transf_act
                        min_rho = sim_rho
                        idx = i

            # if we find topologies: let's check them
            if topo_act and (min_rho < self.topo_threshold):
                self.next_actions = split_action_and_return(observation, self.action_space,
                                                            topo_act.to_vect())

                next_action = next(self.next_actions, None)
                if next_action.as_dict() != {}:
                    idx_chosen = 10000 + idx  # Just as indicator that we chose a topology
                    logging.info(f"Select Topology {idx} with sim_rho of {min_rho}")
                else:
                    logging.info(f"No Topology found. Return Default Action")
                    idx_chosen = -1

                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                return next_action, idx_chosen

            else:
                logging.info(f"No Topology found. Return Default Action")
                return default_action, -1

        ###################################
        # Option 2: Senior method if no Topology was suitable and rho above self.best_action_threshold
        logging.info(
            f" line-{observation.rho.argmax()}d load is {observation.rho.max()}"
        )

        min_rho = observation.rho.max()
        sorted_actions = self.__get_actions(obs=observation)[:self.max_action_sim]

        for k, idx in enumerate(sorted_actions):
            action_vect = self.actions[idx, :]
            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                                    action_vect=action_vect, check_overload=self.check_overload
                                                    )
            if not valid_action:
                continue

            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                logging.info(f"take action {idx}, max-rho to {rho_max}," f" simulation times: {k + 1}")
                idx_chosen = idx
                break

            if rho_max < min_rho:
                # If we have a decrease in rho, we already save the candidate.
                min_rho = rho_max
                idx_chosen = idx

        if idx_chosen:
            self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
            return next_action, idx_chosen

        # If no action was found return default action
        return default_action, -1

    def __collect_action(self, this_directory_path: str, action_space_path: Union[Path, List[Path]]) -> np.ndarray:
        """Check the action space path for the different action set.

        Args:
            this_directory_path: Directory of the submission files
            action_space_path: Optional action space path

        Returns:

        """

        actions = None
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logging.info(f"action_space_path {action_space_path} is a file and will be loaded.")
                actions = np.load(str(action_space_path))
            elif action_space_path.is_dir():
                logging.info(
                    f"action_space_path {action_space_path} is a path. All available action files "
                    f" will be loaded."
                )
                all_action_files = [
                    act for act in os.listdir(action_space_path) if "actions" in act and ".npy" in act
                ]

                if not all_action_files:
                    raise FileNotFoundError("No actions files were found!")

                loaded_files = []
                for act in all_action_files:
                    if "actions" in act and ".npy" in act:
                        loaded_files.append(np.load(action_space_path / act))

                actions = np.concatenate(loaded_files, axis=0)

        elif isinstance(action_space_path, list):
            logging.info(f"action_space_path {action_space_path} is a list containing multiple actions.")
            for act_path in action_space_path:
                if isinstance(act_path, Path):
                    assert act_path.is_file()
                else:
                    os.path.isfile(act_path)
            loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
            actions = np.concatenate(loaded_files, axis=0)
        else:
            raise ValueError(
                f"The action_space_path variable {action_space_path} does neither consist of a single "
                f"action nor of a path where actions can be found."
            )

        return actions

    def __get_actions(self, obs: grid2op.Observation.BaseObservation):
        """ This method conducts depending on the underlying model the action method

        Args:
            obs: Input of the Grid2op Environment

        Returns: action

        """

        if isinstance(self.subset, list):
            model_input = obs.to_vect()[self.subset]
        elif self.subset:
            model_input = obs_to_vect(obs, False)
        else:
            model_input = obs.to_vect()

        if self.scaler:
            model_input = self.scaler.transform(model_input.reshape(1, -1))

        model_input = model_input.reshape((1, -1))

        if isinstance(self.model, tf.keras.models.Sequential):
            # Junior Model: Sequential model
            action_probability = self.model.predict(model_input)
        else:
            # Senior Model: tensorflow functional model:
            action_probability_pre_softmax, _ = self.model.predict(model_input)
            action_probability = tf.nn.softmax(action_probability_pre_softmax).numpy().reshape(-1)
        sorted_actions = action_probability.argsort()[::-1]
        return sorted_actions.reshape(-1)
