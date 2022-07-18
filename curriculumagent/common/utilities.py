"""
In this file all utility methods are collected that are jointly used by the teacher, tutor, junior and senior agent.
The major task of the methods is to communicate with the Grid2Op Environment.

Credit: The methods are the enhanced methods of the original code, see
@https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""



import logging
from copy import deepcopy
from typing import Optional, List, Tuple, Iterator

import numpy as np
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation


def find_best_line_to_reconnect(obs: BaseObservation, original_action: BaseAction) -> BaseAction:
    """Given an observation and action try to reconnect a line by modifying the given original action
    and returning the modified action with the reconnection if possible.

    Args:
        obs: The current observation of the agent.
        original_action: The action the agent is going to take.

    Returns: The modified original_action which tries to reconnect disconnected lines.

    """
    disconnected_lines = np.where(obs.line_status == False)[0]
    if len(disconnected_lines) == 0:
        return original_action
    min_rho = 10
    line_to_reconnect = -1
    for line in disconnected_lines:
        if not obs.time_before_cooldown_line[line]:
            reconnect_array = np.zeros_like(obs.rho, dtype=np.int)
            reconnect_array[line] = 1
            reconnect_action = deepcopy(original_action)
            reconnect_action.update({'set_line_status': reconnect_array})
            if not is_legal(reconnect_action, obs):
                continue
            o, _, _, _ = obs.simulate(reconnect_action)
            if o.rho.max() < min_rho:
                line_to_reconnect = line
                min_rho = o.rho.max()
    if line_to_reconnect != -1:
        reconnect_array = np.zeros_like(obs.rho, dtype=np.int)
        reconnect_array[line_to_reconnect] = 1
        original_action.update({'set_line_status': reconnect_array})
    return original_action


def is_legal(action: BaseAction, obs: BaseObservation) -> bool:
    """Return true if the given action is valid under the current given observation.

    Args:
        action: The action to check for.
        obs: The current observation of the environment.

    Returns: Whether the given action is valid/legal or not.

    """
    action_dict = action.as_dict()

    if action_dict == {}:
        return True

    topo_action_type = list(action_dict.keys())[0]
    legal_act = True
    # Check substations:
    if topo_action_type == 'set_bus_vect' or topo_action_type == 'change_bus_vect':
        substations = [int(sub) for sub in action.as_dict()[topo_action_type]['modif_subs_id']]

        for substation_to_operate in substations:
            if obs.time_before_cooldown_sub[substation_to_operate]:
                # substation is cooling down
                legal_act = False
            # Check lines:
            for line in [eval(key) for key, val in
                         action.as_dict()[topo_action_type][str(substation_to_operate)].items()
                         if 'line' in val['type']]:
                if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                    # line is cooling down, or line is disconnected
                    legal_act = False
    elif topo_action_type == 'set_line_status':

        lines = [int(line) for line in action.as_dict()[topo_action_type]['connected_id']]
        for line in lines:
            if obs.time_before_cooldown_line[line]:
                legal_act = False

    return legal_act


def array2action(action_space: ActionSpace,
                 action_vect: np.ndarray, reconnect_vect: Optional[np.ndarray] = None) -> BaseAction:
    """Given an action space, convert the given action_vect into a grid2op Action.

    Args:
        action_space: The action space to use.
        action_vect: Main action to read from. Should mostly contain topology actions.
        reconnect_vect: Maybe set to a vector of lines, mostly for reconnect them after failure.

    Returns: The matching grid2op action.

    """
    action = action_space.from_vect(action_vect)
    if reconnect_vect is not None:
        action.update({'set_line_status': reconnect_vect})
    return action


def get_from_dict_set_bus(original: dict) -> dict:
    """ Convert action from dictionary based on BaseAction.as_dict() to a dictionary that can be used
    as input for the action space.

    Args:
        original:

    Returns: Dictionary with set_bus action

    """
    dict_act = {"lines_or_id": [], "lines_ex_id": [], "loads_id": [], "generators_id": []}
    for key, value in original.items():
        for old, new in [('line (origin)', "lines_or_id"),
                         ('line (extremity)', "lines_ex_id"),
                         ('load', "loads_id"),
                         ('generator', "generators_id")]:
            if old == original[key]["type"]:
                dict_act[new].append((int(key), int(value["new_bus"])))

    return {"set_bus": dict_act}


def extract_action_set_from_actions(action_space: ActionSpace, action: np.ndarray,
                                    reconnect_vect: Optional[np.ndarray] = None) -> List[BaseAction]:
    """ Method to separate multiple substation actions into single actions.

    This method is necessary to ensure that the tuple and triple actions are in accordance to
    the Grid2Op rules.

    Args:
        action_space: action space of Grid2Op environment
        action: single action from numpy array
        reconnect_vect: Set of lines that can be reconnected

    Returns:

    """
    action_set = []

    # Check if do nothing action:
    if not action.any():
        return [action_space({})]

    # Convert into action:
    act_dict = array2action(action_space=action_space,
                            action_vect=action,
                            reconnect_vect=reconnect_vect).as_dict()
    if list(act_dict.keys())[0] == 'set_bus_vect':
        act_t = act_dict['set_bus_vect']

        # Get sub-ids
        changed_sub_ids = act_t['modif_subs_id']

        if len(changed_sub_ids) > 1:
            # Collect single action
            for sub_id in changed_sub_ids:
                action_set.append(action_space(get_from_dict_set_bus(act_t[sub_id])))
            return action_set
        else:
            return [array2action(action_space, action)]  #

    if list(act_dict.keys())[0] == 'change_bus_vect':
        # We have an old action path with only change_bus actions
        # These actions are assumed to be unitary, thus we only return this action:
        act_t = act_dict['change_bus_vect']
        if len(act_t['modif_subs_id']) == 1:
            return [array2action(action_space=action_space,
                                 action_vect=action,
                                 reconnect_vect=reconnect_vect)]
        else:
            raise NotImplementedError("Multiple substations were modified in the change_bus action. This is not yet "
                                      "implemented in the tuple and triple approach. Please one use set_bus actions "
                                      "or unitary change_bus actions")
    else:
        logging.warning("Attention, a action was provided which could not be accounted for by the "
                        "extract_action_set_from_actions method.")
        return [array2action(action_space=action_space,
                             action_vect=action,
                             reconnect_vect=reconnect_vect)]


def split_action_and_return(obs: BaseObservation, action_space: ActionSpace, action: np.ndarray) \
        -> Iterator[BaseAction]:
    """Split an action with potentially multiple affected stations and return them sequentially as a generator/iterator.

    Depending on the input, the method either executes the numpy array as a unitary step, or
    if the input requires multiple steps (i.e. in case of a tuple or tripple action) the method
    does multiple steps.

    Note that if multiple steps are computed only the last observation is return. Further, the
    reward then consists of the cummulative reward. As Example: The action is a tripple action. Thus, the
    reward is the cummulative reward of all three steps.

    All actions are checked for line reconnect.

    Args:
        obs: Current Observation
        action: Teacher Action that can either be a unitary, tuple or tripple action

    Returns: The next best action to execute.
    """

    # Special case: Do nothing
    if not action.any():
        yield action_space.from_vect(action)
        return

    # First extract action:
    split_actions = extract_action_set_from_actions(action_space, action)
    # Now simulate through all actions:
    for _ in range(len(split_actions)):
        # Iterate through remaining actions and choose the best one to execute next
        obs_min = np.inf
        best_choice = None

        for act in split_actions:
            act_plus_reconnect = find_best_line_to_reconnect(obs, act)
            if not is_legal(act_plus_reconnect, obs):
                continue

            obs_, _, done, _ = obs.simulate(act_plus_reconnect)
            if obs_.rho.max() < obs_min and done == False:
                best_choice = act_plus_reconnect
                obs_min = obs_.rho.max()

        if best_choice is None:
            logging.info("No action was suitable, take empty action. ")
            best_choice = find_best_line_to_reconnect(obs, action_space({}))

        # Assert whether a reconnection of the lines might be
        yield best_choice
        if best_choice in split_actions:
            split_actions.remove(best_choice)


def split_and_execute_action(env: BaseEnv,
                             action: np.ndarray) -> Tuple[BaseObservation, float, bool, dict]:
    """Split and execute an action with potentially multiple affected stations.

    Depending on the input, the method either executes the numpy array as a unitary step, or
    if the input requires multiple steps (i.e. in case of a tuple or tripple action) the method
    does multiple steps.

    Note that if multiple steps are computed only the last observation is return. Further, the
    reward then consists of the cummulative reward. As Example: The action is a tripple action. Thus, the
    reward is the cummulative reward of all three steps.

    All actions are checked for line reconnect.

    Args:
        env: Grid2Op Environment
        obs: Current Observation
        action: Teacher Action that can either be a unitary, tuple or tripple action

    Returns: Output of the env.step() function consisting of an Observation, (cumulative) Reward, a Done statement
    and info.

    """

    # First extract action:
    split_actions = extract_action_set_from_actions(env.action_space, action)
    # Now simulate through all actions:
    cum_rew = 0
    done = False
    obs = env.get_obs()
    info = {}
    for _ in range(len(split_actions)):
        # Iterate through remaining actions and choose the best one to execute next
        obs_min = np.inf
        best_choice = find_best_line_to_reconnect(obs, env.action_space({}))

        chosen_act = None
        for act in split_actions:
            act_plus_reconnect = find_best_line_to_reconnect(obs, act)
            if not is_legal(act_plus_reconnect, obs):
                continue

            obs_, _, done, _ = obs.simulate(act_plus_reconnect)
            if obs_.rho.max() < obs_min and done is False:
                best_choice = act_plus_reconnect
                chosen_act = act
                obs_min = obs_.rho.max()

        # Assert whether a reconnection of the lines might be

        obs, reward, done, info = env.step(best_choice)
        cum_rew += reward
        if done:
            break
        if chosen_act:
            split_actions.remove(chosen_act)

    return obs, cum_rew, done, info
