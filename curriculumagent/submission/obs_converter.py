"""This file contains features/mappings/extractions for training."""
import logging
from typing import Optional

import grid2op
import numpy as np


def obs_to_vect(obs: grid2op.Observation.BaseObservation, connectivity: bool = False) -> np.ndarray:
    """Method to convert only a subset of the observation to a vector.

    Args:
        obs: Original observation of Grid2Op.
        connectivity: Indicator, whether the connectivity matrix should be saved as well.

    Returns:
        Vector of the observation.

    """
    features = [
        # Timestamp Features
        [obs.month, obs.day, obs.hour_of_day, obs.minute_of_hour, obs.day_of_week],
        # Generation and Load
        obs.gen_p,
        obs.gen_q,
        obs.gen_v,
        obs.load_p,
        obs.load_q,
        obs.load_v,
        # Raw line values
        obs.p_or,
        obs.q_or,
        obs.v_or,
        obs.a_or,
        obs.p_ex,
        obs.q_ex,
        obs.v_ex,
        obs.a_ex,
        obs.rho,
        obs.line_status,
        obs.timestep_overflow,
        # Bus information
        obs.topo_vect,
        # cool downs:
        obs.time_before_cooldown_line,
        obs.time_before_cooldown_sub,
        # maintenance
        obs.time_next_maintenance,
        obs.duration_next_maintenance,
    ]

    if connectivity:
        features.append(obs.connectivity_matrix().reshape(-1))

    return np.concatenate(features, dtype=np.float32)


def vect_to_dict(
        vect: np.ndarray, examplary_obs: grid2op.Observation.BaseObservation, connectivity: bool = False
) -> dict:
    """Method that converts the vector of the obs_to_vect method to a dictionary.
    Note that  for this one we require an observation of the environment in order to gather the correct information.

    Args:
        vect: Vector of the obs subset
        examplary_obs: One Grid2Op environment to get the correct lengths.
        connectivity: Whether to return the connectivity or not. This is only possible, if
        connectivity matrix to begin with was saved before.

    Returns:
        A dictionary of the observation.

    """
    if not isinstance(vect, np.ndarray):
        raise TypeError("vect input does not have the correct type")
    if not isinstance(examplary_obs, grid2op.Observation.BaseObservation):
        raise TypeError("examplary_obs input does not have the correct type. Please enter the observation "
                        "of the grid2op environment")

    assert len(vect.shape) < 2, "The dimensions of the vect input are not correct. Should be a vector"

    out = {
        # The first 5 are allways the same:
        "month": vect[0],
        "day": vect[1],
        "hour_of_day": vect[2],
        "minute_of_hour": vect[3],
        "day_of_week": vect[4],
    }
    i = 5
    obs_json = examplary_obs.to_json()

    for k in [
        "gen_p",
        "gen_q",
        "gen_v",
        "load_p",
        "load_q",
        "load_v",
        "p_or",
        "q_or",
        "v_or",
        "a_or",
        "p_ex",
        "q_ex",
        "v_ex",
        "a_ex",
        "rho",
        "line_status",
        "timestep_overflow",
        "topo_vect",
        "time_before_cooldown_line",
        "time_before_cooldown_sub",
        "time_next_maintenance",
        "duration_next_maintenance",
    ]:
        out[k] = vect[i: i + len(obs_json[k])]
        i += len(obs_json[k])

    if connectivity:
        if np.sqrt(len(vect[i:])) % 1 == 0:
            c_m = vect[i:].reshape(examplary_obs.connectivity_matrix().shape)
            out["connectivity_matrix"] = c_m
        else:
            logging.warning("The connectivity Matrix is not quadratic. Thus, it is not added to the dictionary")

    return out