from pathlib import Path

import grid2op
import numpy as np
import pytest
from grid2op.Environment import Environment
from grid2op.Reward import RedispReward, L2RPNSandBoxScore
from lightsim2grid import LightSimBackend

from curriculumagent.teacher.teachers.tuple_triple_teacher import run_multi_teacher


@pytest.mark.skip(reason="takes too long")
@pytest.mark.ultra_slow
@pytest.mark.slow
def test_tuple_teacher_run():
    """
    Quickly test the tutor by running it for a small(300) number of steps in the test environment
    and comparing it against previous/good runs.
    """
    test_flag = True
    seed = 42
    reward = RedispReward
    other_rewards = {"grid_operation_cost": L2RPNSandBoxScore}

    test_env_path = Path(__file__).parent.parent / "training_data_track1"
    env: Environment = grid2op.make(
        test_env_path, test=test_flag, backend=LightSimBackend(), reward_class=reward, other_rewards=other_rewards
    )
    np.random.seed(seed)
    env.seed(seed)

    run_multi_teacher(
        Path("./exp.csv"), env_name_path=test_env_path, n_episodes=1, top_k=20, limit_chronics=1, jobs=1, seed=seed
    )
