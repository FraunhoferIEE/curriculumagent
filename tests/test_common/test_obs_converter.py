import logging
import math

import grid2op.Action
import numpy as np
import pytest
import torch

from curriculumagent.common.obs_converter import obs_to_vect, vect_to_dict

class TestConvertObs:
    """
    In this test class we test both the method
    """

    def test_obs_to_vect(self, test_env):
        """
        Testing, whether the subset is collected correctly
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, False)
        # Should only be a subset len of 1221
        assert len(obs.to_vect()) > len(sub_obs)
        assert np.all(obs.to_vect()[1:1222] == sub_obs)

    def test_obs_to_vect_connectivity(self, test_env):
        """
        Testing, if the connectivity matrix is correctly appended
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, True)

        # Assert, whether the coneectivity matrix is correctly pulled:
        con_mat = obs.connectivity_matrix().reshape(-1)
        assert np.all(con_mat == sub_obs[1221:])
        mt_dim = int(np.sqrt(len(sub_obs[1221:])))
        assert np.all(obs.connectivity_matrix() == sub_obs[1221:].reshape(mt_dim, mt_dim))

    def test_old_chosen(self, test_env):
        """
        Testing, whether the old chosen parameter return the same as the new obs_to_vect method
        """
        # Taken from the ppo_environment
        chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        chosen += (
                list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        )
        chosen = np.asarray(chosen, dtype=int) - 1
        # get sub_obs
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, False)
        old_chosen_obs = obs.to_vect()[chosen]
        assert np.all(sub_obs == old_chosen_obs)

    def test_vect_to_dict_error(self, test_env):
        """
        Testing, whether the dictionary is correctly
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, True)

        with pytest.raises(TypeError):
            vect_to_dict({"wrong": "input"}, obs, False)

        with pytest.raises(TypeError):
            vect_to_dict(sub_obs, sub_obs, False)

        with pytest.raises(AssertionError):
            vect_to_dict(sub_obs.reshape(-1, 3), obs, False)

    def test_vect_to_dict(self, test_env):
        """
        Testing, whether the dictionary is correctly
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, False)
        dict_out = vect_to_dict(sub_obs, obs, False)

        for key in [
            "month", "day", "hour_of_day", "minute_of_hour", "day_of_week",
            "gen_p", "gen_q", "gen_v", "load_p", "load_q", "load_v", "p_or", "q_or",
            "v_or", "a_or", "p_ex", "q_ex", "v_ex", "a_ex", "rho", "line_status",
            "timestep_overflow", "topo_vect", "time_before_cooldown_line",
            "time_before_cooldown_sub", "time_next_maintenance", "duration_next_maintenance"
        ]:
            assert np.all(obs.to_json()[key] == dict_out[key])

    def test_vect_to_dict_connectivity_warning(self, caplog, test_env):
        """
        Testing, when the connectivity matrix is not added to a full extend, that a warning is raised
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, True)
        with caplog.at_level(logging.WARNING):
            exit_dict = vect_to_dict(sub_obs[:-2], obs, True)
        # Test, if correctly warned
        assert "The connectivity Matrix is not quadratic" in caplog.text
        assert "connectivity_matrix" not in exit_dict.keys()

    def test_vect_to_dict_connectivity(self, test_env):
        """
        Test if the connectivity matrix ist constructed correctly
        """
        obs = test_env.get_obs()
        sub_obs = obs_to_vect(obs, True)
        dict_out = vect_to_dict(sub_obs, obs, True)
        assert np.all(dict_out["connectivity_matrix"] == obs.connectivity_matrix())