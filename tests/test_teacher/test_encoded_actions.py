import grid2op

from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction


def test_unitary_action_encodings():
    """Test the encoding and decoding on all unitary topology actions of a simple environment."""
    env = grid2op.make("l2rpn_case14_sandbox")

    all_actions = env.action_space.get_all_unitary_topologies_set(env.action_space)
    for action in all_actions:
        encoded = EncodedTopologyAction(action)
        decoded = encoded.to_action(env)
        if decoded != action:
            print(f"{decoded} != {action}")
        assert decoded == action


def test_do_nothing_action():
    """Test the encoding and decoding on all unitary topology actions of a simple environment."""
    env = grid2op.make("l2rpn_case14_sandbox")

    do_nothing = env.action_space()
    encoded = EncodedTopologyAction(do_nothing)
    assert encoded.data == "0"
    assert encoded.to_action(env) == do_nothing

    # Another shortcut for encoding the do nothing action
    assert EncodedTopologyAction(None).data == "0"
