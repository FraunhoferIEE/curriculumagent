"""This file consist of teh EncodedTopologyAction class, which is used to convert the teacher_experience of the teacher
to a more cache friendly way.

"""

import base64
import zlib
from typing import Optional

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv


class EncodedTopologyAction:
    """A topology/set_bus action encoded as a zlib-compressed base64 string to save it more easily in a csv.

    Note:
        This only encodes set_bus actions, all other action types like redispatch are currently ignored!
    """

    def __init__(self, action: Optional[BaseAction]):
        """Create and encode the given action to a new instance of this object.

        Args:
            action: The Grid2Op action to encode. If set to None, the do nothing action will be encoded.

        Return:
            None.
        """
        self.data: str = self.encode_action(action)

    def to_action(self, env: BaseEnv) -> BaseAction:
        """Decode the action back to a Orid2Op action.

        Args:
            env: The environment this action belongs to.

        Returns:
            The Grid2Op action usable by an agent.
        """
        return self.decode_action(self.data, env)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return self.data

    @staticmethod
    def encode_action(action: Optional[BaseAction]) -> str:
        """Pack a set_bus action into a base64 string to make it hashable and more efficient to save in a .csv file.

        Args:
            action: The Grid2Op action to encode. If set to None, the do nothing action will be encoded.

        Returns:
            An utf8 string containing a base64 encoded representation of the change_bus action.
        """
        # Check if the given action can be encoded.
        if not action:
            return "0"
        assert not (
                action._modif_inj
                and action._modif_change_bus
                and action._modif_set_status
                and action._modif_change_status
                and action._modif_redispatch
                and action._modif_storage
                and action._modif_curtailment
                and action._modif_alarm
        ), "Given action type can be encoded"
        # Special case: Empty action -> Encode with 0
        if not action._modif_set_bus:
            return "0"

        packed_action = zlib.compress(action._set_topo_vect, level=1)
        encoded_action = base64.b64encode(packed_action)
        return encoded_action.decode("utf-8")

    @staticmethod
    def decode_action(act_string: str, env: BaseEnv) -> BaseAction:
        """Unpack the previously encoded string to na action for the given environment.

        Args:
            act_string: The string containing the encoded action.
            env: The environment this action belongs to.

        Returns:
            The Grid2Op action usable by an agent.
        """
        unpacked_act: BaseAction = env.action_space()
        # Special case: Empty action
        if act_string == "0":
            return unpacked_act
        decoded = base64.b64decode(act_string.encode("utf-8"))
        unpacked = np.frombuffer(zlib.decompress(decoded), dtype=np.int32)
        unpacked_act._set_topo_vect = unpacked
        unpacked_act._modif_set_bus = True
        return unpacked_act
