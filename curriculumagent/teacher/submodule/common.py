"""This file constist of common methods used by multiple teacher agents.

Most methods are based on the original code of EI Innovation Lab, Huawei Cloud, Huawei Technologies.

"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation

from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction


def save_sample_new(
        save_path: Path,
        best_actions: List[Tuple[float, EncodedTopologyAction]],
        obs: BaseObservation,
        obs_: BaseObservation,
        top_k: int,
):
    """Save the given action taken in the given environment for further investigation and action space reduction.

    Args:
        save_path: The path to the csv file where the teacher_experience should be saved/appended.
        best_actions: The best actions that could be taken.
        obs: Observation before the action.
        obs_: Observation after taking the action.
        top_k: How many of the best_actions should be saved.

    Returns:
        None.
    """
    rho_max_old = obs.rho.max()
    rho_max_new = obs_.rho.max()
    rho_improvement: float = best_actions[0][0]

    best_actions = best_actions[:top_k]
    best_action = best_actions[0][1]
    best_actions_only: List[EncodedTopologyAction] = [action[1] for action in best_actions]
    n_actions = len(best_actions_only)
    for i in range(top_k - n_actions):
        best_actions_only.append(EncodedTopologyAction(None))
    if len(best_actions_only) != top_k:
        print(f"Didn't match: {len(best_actions_only)} != {top_k}")
        raise AssertionError()

    rho_improvement_only = [action[0] for action in best_actions]

    data = np.concatenate(
        (
            np.array((rho_max_old, rho_max_new, rho_improvement, best_action)),
            np.array(best_actions_only),
            np.array(rho_improvement_only),
        )
    ).reshape([1, -1])

    # Save to a default path if save_path only directory
    if save_path.is_dir():
        save_path = save_path / "teacher_experience.csv"

    if not save_path.exists():
        column_names = ["rho_max_old", "rho_max_new", "rho_improvement", "best_action"]
        for i in range(len(best_actions_only)):
            column_names.append(f"top_{i}_action")
        for i in range(len(best_actions_only)):
            column_names.append(f"top_{i}_ri")
        # Write header when first writing to file
        data_df = pd.DataFrame(data, columns=column_names)
        data_df.to_csv(save_path, index=False, header=True, mode="a")
    else:
        # Else just append to file
        data_df = pd.DataFrame(data)
        data_df.to_csv(save_path, index=False, header=False, mode="a")


station_id_maps = {}


def affected_substations(action: BaseAction) -> List[int]:
    """Optimizing routine to get all the affected substations of an action.

    Args:
        action: The action to list the affected stations from.

    Returns: A tuple of affected station ids.
    """
    global station_id_maps
    if action.env_name not in station_id_maps:
        station_id_maps[action.env_name] = make_station_id_lookup_map(action.sub_info)
    station_id_map = station_id_maps[action.env_name]
    topo_vect = action._set_topo_vect  # noqa
    affected_sub_ids = np.sort(np.unique(station_id_map[topo_vect > 0]))
    return list(affected_sub_ids)


def make_station_id_lookup_map(sub_info: np.ndarray) -> np.ndarray:
    """Create lookup table for matching station objects in _set_topo_vect to substation ids.

    Args:
        sub_info: The sub_info vector usually taken from a Grid2Op action.

    Returns:
        A numpy array that maps from _set_topo_vect index to substation id
    """
    n_lines = np.sum(sub_info)
    lookup_map = np.full(n_lines, -1, dtype=int)
    i = 0
    for sub_id, n_obj in enumerate(sub_info):
        for _ in range(n_obj):
            lookup_map[i] = sub_id
            i += 1
    return lookup_map
