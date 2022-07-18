import lzma
import pickle
from pathlib import Path

import grid2op
import numpy as np
from grid2op.Environment import Environment
from grid2op.Reward import RedispReward, L2RPNSandBoxScore
from lightsim2grid import LightSimBackend

from curriculumagent.tutor.tutors import original_tutor


def test_tutor_original_actions(test_paths_env):
    """
    Quickly test the tutor by running it for a small(300) number of steps in the test environment
    and comparing it against previous/good runs.
    """
    # Subset for observations that are required, given that the obs change for the different versions.
    chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
    chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
    chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))


    test_flag = True
    seed = 42
    reward = RedispReward
    other_rewards = {'grid_operation_cost': L2RPNSandBoxScore}
    t_path_env, t_path_action = test_paths_env

    env: Environment = grid2op.make(str(t_path_env.resolve()), test=test_flag,
                                    backend=LightSimBackend(), reward_class=reward, other_rewards=other_rewards)
    np.random.seed(seed)
    env.seed(seed)
    agent = original_tutor.Tutor(env.action_space, old_actionspace_path=t_path_action)

    done = False
    reward = env.reward_range[0]
    obs = env.reset()
    experience = []
    n_steps = 300
    for i in range(n_steps):
        action = agent.act(obs, 0)
        obs, reward, done, info = env.step(action)
        experience.append((action, obs.to_vect()))
        if done:
            assert False, f"Environment stopped early/ agent couldn't solve {n_steps} steps"

    good_exp_file = Path(__file__).parent.parent / 'data/tutor_gold_sample.pkl.xz'
    with lzma.open(good_exp_file, "rb") as f:
        good_experience = pickle.load(f)

    # Use this to save new gold/good data when the tutor performs well
    save_data = False
    if save_data:
        with lzma.open(good_exp_file, "wb") as f:
            pickle.dump(experience, f, pickle.HIGHEST_PROTOCOL)

    for i, (action, obs) in enumerate(experience):
        good_action, good_obs = good_experience[i]
        assert action == good_action, f'action {action} at {i} does match'
        # Given that for Grid2op 1.7.1 the observations are larger then the previous ones, we have
        # to update the "good" observations.
        assert (np.round(obs[chosen],4) == np.round(good_obs[chosen],4)).all(), f'observation {obs} at {i} does match'


def test_tutor_rlagent_actions(test_paths_env):
    """
    Run the tutor with rlagent action space and make sure it "works".
    TODO: Could use a gold sample aswell.
    """
    test_flag = True
    seed = 42
    reward = RedispReward
    other_rewards = {'grid_operation_cost': L2RPNSandBoxScore}
    t_path_env, t_path_action = test_paths_env

    env: Environment = grid2op.make(str( t_path_env.resolve()), test=test_flag,
                                    backend=LightSimBackend(), reward_class=reward, other_rewards=other_rewards)
    np.random.seed(seed)
    env.seed(seed)
    agent = original_tutor.Tutor(env.action_space, action_space_file=Path(__file__).parent.parent /
                                                            'data/rlagent_unitary_actions.npy')

    done = False
    reward = env.reward_range[0]
    obs = env.reset()
    experience = []
    n_steps = 100
    for i in range(n_steps):
        action = agent.act(obs, 0)
        obs, reward, done, info = env.step(action)
        experience.append((action, obs.to_vect()))
        if done:
            assert False, f"Environment stopped early/ agent couldn't solve {n_steps} steps"
