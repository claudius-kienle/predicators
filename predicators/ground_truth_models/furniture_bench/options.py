"""Ground-truth options for the furniture bench environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class FurnitureBenchGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the furniture bench environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"furniture_bench"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        # top_bun_type = types["top_bun"]
        # bottom_bun_type = types["bottom_bun"]
        # cheese_type = types["cheese"]
        robot_type = types["robot"]
        table_type = types["table"]
        part_type =  types["part"]

        object_type = types["object"]

        # Predicates
        HoveredAbove = predicates["HoveredAbove"]
        OnTable = predicates["OnTable"]
        Touching = predicates["Touching"]
        Screwable = predicates["Screwable"]
        Holding = predicates["Holding"]
        Aligned = predicates["Aligned"]
        Assembled = predicates["Assembled"]
        GripperAround = predicates["GripperAround"]
        def _dummy_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            return True

        HoverAbovePart = ParameterizedOption(
            "HoverAbovePart",
            types=[robot_type, part_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        SetGripperAroundPart = ParameterizedOption(
            "SetGripperAroundPart",
            types=[robot_type, part_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        MoveLinearUp = ParameterizedOption(
            "MoveLinearUp",
            types=[robot_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        OpenGripper = ParameterizedOption(
            "OpenGripper",
            types=[robot_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        CloseGripper = ParameterizedOption(
            "CloseGripper",
            types=[robot_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        AlignOrientationForAssembly = ParameterizedOption(
            "AlignOrientationForAssembly",
            types=[robot_type, part_type, part_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        ScrewTouchingPartsTogether = ParameterizedOption(
            "ScrewTouchingPartsTogether",
            types=[robot_type, part_type, part_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        MoveLinearDownUntilTouching = ParameterizedOption(
            "MoveLinearDownUntilTouching",
            types=[robot_type, part_type, part_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal)

        return {HoverAbovePart, SetGripperAroundPart, MoveLinearUp, OpenGripper, 
                CloseGripper, AlignOrientationForAssembly, ScrewTouchingPartsTogether, 
                MoveLinearDownUntilTouching}

    @classmethod
    def _dummy_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            raise NotImplementedError()

        return policy

