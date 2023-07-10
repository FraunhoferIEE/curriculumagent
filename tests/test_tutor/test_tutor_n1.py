"""
Test suite
"""
import os
import random
from pathlib import Path

import grid2op
import numpy as np
import pytest

from lightsim2grid import LightSimBackend
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor
from curriculumagent.tutor.tutors.n_minus_one_tutor import NminusOneTutor
class TestTutorNminus1:
    """
    Test suite for General Tutor
    """

    def test_if_running_works(self, test_paths_env, test_action_paths):
        """
        Test, whether the N-1 Path is run:
        """

        backend = LightSimBackend()

        kwargs = {"opponent_budget_per_ts": 0, "opponent_init_budget": 0}

        # Deactivate the Adversarial agent:
        env_path, _ = test_paths_env
        test_env = grid2op.make(dataset=env_path, backend=backend, **kwargs)

        # First run General Tutor:
        single_path, _, _ = test_action_paths
        tutor = GeneralTutor(
            action_space=test_env.action_space,
            action_space_file=single_path,
            do_nothing_threshold=0.9,
            best_action_threshold=1.00,
            return_status=True,
        )

        test_env.seed(42)
        test_env.set_id(1)
        obs = test_env.reset()
        obs.rho[0] = 0.95

        act = tutor.act(observation=obs, reward=0, done=False)
        assert act.as_dict()["set_bus_vect"]["modif_subs_id"] == ["12"]
        tutor = NminusOneTutor(
            action_space=test_env.action_space,
            action_space_file=single_path,
            do_nothing_threshold=0.9,
            best_action_threshold=1.00,
            rho_greedy_threshold=0.95,
            lines_to_check=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
            return_status=True,
        )

        test_env.seed(42)
        test_env.set_id(1)
        obs = test_env.reset()
        obs.rho[0] = 0.95

        act_n1 = tutor.act(observation=obs, reward=0, done=False)
        assert act_n1.as_dict()["set_bus_vect"]["modif_subs_id"] == ["30"]

    def test_multiple_actions_idx(self,test_env, test_action_single):
        """
        Testing, whether multiple actions are selected:
        """
        single_path1, single_path2 = test_action_single
        tutor = NminusOneTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path1, single_path2],
            do_nothing_threshold=0.9,
            best_action_threshold=1.00,
            rho_greedy_threshold=0.95,
            lines_to_check=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
            return_status=True,
        )

        random.seed(42)
        np.random.seed(42)
        test_env.seed(42)
        test_env.set_id(1)
        obs = test_env.reset()
        while obs.rho.max() <= 1.0:
            obs,_,_,_ = test_env.step(test_env.action_space({}))

        # First dictionary:
        array_out, idx = tutor.act_with_id(observation=obs)
        assert idx == 6 # first subset of actions


        # Now switch the values:
        tutor = NminusOneTutor(
            action_space=test_env.action_space,
            action_space_file=[single_path2,single_path1],
            do_nothing_threshold=0.9,
            best_action_threshold=1.00,
            rho_greedy_threshold=0.95,
            lines_to_check=[45, 56, 0, 9, 13, 14, 18, 23, 27, 39],
            return_status=True,
        )

        test_env.seed(42)
        test_env.set_id(1)
        obs = test_env.reset()
        while obs.rho.max() <= 1.0:
            obs,_,_,_ = test_env.step(test_env.action_space({}))

        # First dictionary:
        array_out, idx = tutor.act_with_id(observation=obs)
        assert idx == 1 # First subset of actions


