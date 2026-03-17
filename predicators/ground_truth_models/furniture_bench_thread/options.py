"""Ground-truth options for the furniture_bench_thread environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class FurnitureBenchThreadGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the furniture_bench_thread environment.

    Compared to FurnitureBenchGroundTruthOptionFactory:
    - Adds a ``thread`` type; options that target a thread accept thread objects.
    - All options carry an explicit robot argument.
    - ``AlignPartForThread`` / ``MoveLinearDownUntilTouching`` /
      ``ScrewTouchingPartsTogether`` / ``OpenGripper`` take (robot, part, top,
      thread) matching the PDDL action signatures.
    - ``HoverAboveThread`` is new.
    - ``MoveDownAroundPart`` replaces ``SetGripperAroundPart``.
    """

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"furniture_bench_thread"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Types
        robot_type  = types["robot"]
        part_type   = types["part"]
        thread_type = types["thread"]

        def _dummy_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            return True

        # hover-above-part-or-thread(?r ?p)
        HoverAbovePartOrThread = ParameterizedOption(
            "HoverAbovePartOrThread",
            types=[part_type],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # move-down-around-part(?r ?p)
        MoveDownAroundPart = ParameterizedOption(
            "MoveDownAroundPart",
            types=[part_type],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # move-linear-up(?r)
        MoveLinearUp = ParameterizedOption(
            "MoveLinearUp",
            types=[],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # close-gripper(?r ?p)
        CloseGripper = ParameterizedOption(
            "CloseGripper",
            types=[],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # open-gripper(?r ?p ?top ?t)
        OpenGripper = ParameterizedOption(
            "OpenGripper",
            types=[],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # align-part-for-thread(?r ?p ?top ?t)
        AlignOrientationForAssembly = ParameterizedOption(
            "AlignOrientationForAssembly",
            types=[part_type, thread_type],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # move-linear-down-until-touching(?r ?p ?top ?t)
        MoveLinearDownUntilTouching = ParameterizedOption(
            "MoveLinearDownUntilTouching",
            types=[part_type, thread_type],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        # screw-touching-parts-together(?r ?p ?top ?t)
        ScrewTouchingPartsTogether = ParameterizedOption(
            "ScrewTouchingPartsTogether",
            types=[part_type, thread_type],
            params_space=Box(0, 1, (0,)),
            policy=cls._dummy_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_dummy_terminal,
        )

        return {
            HoverAbovePartOrThread,
            MoveDownAroundPart,
            MoveLinearUp,
            CloseGripper,
            OpenGripper,
            AlignOrientationForAssembly,
            MoveLinearDownUntilTouching,
            ScrewTouchingPartsTogether,
        }

    @classmethod
    def _dummy_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            raise NotImplementedError()

        return policy
