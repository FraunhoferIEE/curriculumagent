import grid2op
import numpy as np
import pytest
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv

from curriculumagent.teacher.submodule.common import affected_substations


def test_affected_substations():
    env: BaseEnv = grid2op.make(dataset="l2rpn_case14_sandbox")
    action_space = env.action_space

    # Empty action -> No affected substations
    act: BaseAction = action_space()
    subs = affected_substations(act)
    assert subs == [], "Do Nothing Action has no affected substations"

    # Unitary action -> One affected substations
    act: BaseAction = action_space()
    act.set_bus = [(0, 2)]
    subs = affected_substations(act)
    assert subs == [0], "Switching one line affects one substation"

    # Switch every object to busbar 2 -> Should affect all substations
    act: BaseAction = action_space()
    for obj in range(act.dim_topo):
        act.set_bus = [(obj, 2)]  # perform the desired modification
    assert affected_substations(act) == list(
        range(env.n_sub)
    ), "When switching every line we should effect all substations"


def test_affected_substations_comparison():
    env: BaseEnv = grid2op.make(dataset="l2rpn_case14_sandbox")
    action_space = env.action_space

    # Generate random actions and compare it with as_dict
    act: BaseAction = action_space()
    for _ in range(100):
        act._set_topo_vect = np.random.choice([0, 1, 2], size=env.dim_topo)  # noqa
        affected_real = sorted([int(v) for v in act.as_dict()["set_bus_vect"]["modif_subs_id"]])
        assert affected_substations(act) == affected_real

