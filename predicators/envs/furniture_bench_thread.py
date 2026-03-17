"""FurnitureBench environment variant with explicit thread objects and robot
arguments in predicates, designed for thread-based assembly tasks (e.g.
screwing table legs into threaded holes in a table top).

Predicate differences vs FurnitureBenchEnv:
- Thread is a first-class type; aligned/touching are part-thread, not part-part.
- All robot-state predicates carry an explicit robot argument.
- No OnTable or Screwable; adds HasThread, Occupied, HoveringAboveThread,
  AroundPart, GripperOpen, ArmRaised.
"""

import copy
from typing import Callable, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Observation, Predicate, State, Task, Type, Video


class FurnitureBenchThreadEnv(BaseEnv):
    """FurnitureBench environment for thread-based assembly.

    Uses explicit thread objects and robot arguments, matching the PDDL domain
    used by the furniture-bench task planner (fb-furniture-bench-thread).
    """

    # Types
    _object_type = Type("object", [])
    _part_type   = Type("part",   [], _object_type)
    _thread_type = Type("thread", [], _object_type)
    _robot_type  = Type("robot",  [], _object_type)
    _table_type  = Type("table",  [], _object_type)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # -- structural predicates ----------------------------------------
        self._HasThread = Predicate(
            "has_thread",
            [self._part_type, self._thread_type],
            self._has_thread_holds,
        )
        self._Occupied = Predicate(
            "occupied",
            [self._thread_type],
            self._occupied_holds,
        )

        # -- geometric/placement predicates --------------------------------
        self._Aligned = Predicate(
            "aligned",
            [self._part_type, self._thread_type],
            self._aligned_holds,
        )
        self._Touching = Predicate(
            "touching",
            [self._part_type, self._thread_type],
            self._touching_holds,
        )

        # -- robot/part interaction predicates -----------------------------
        self._Holding = Predicate(
            "holding",
            [self._robot_type, self._part_type],
            self._holding_holds,
        )
        self._AroundPart = Predicate(
            "around_part",
            [self._robot_type, self._part_type],
            self._around_part_holds,
        )
        self._HoveringAbovePart = Predicate(
            "hovering_above_part",
            [self._robot_type, self._part_type],
            self._hovering_above_part_holds,
        )
        self._HoveringAboveThread = Predicate(
            "hovering_above_thread",
            [self._robot_type, self._thread_type],
            self._hovering_above_thread_holds,
        )

        # -- robot state predicates ----------------------------------------
        self._GripperOpen = Predicate(
            "gripper_open",
            [self._robot_type],
            self._gripper_open_holds,
        )
        self._ArmRaised = Predicate(
            "arm_raised",
            [self._robot_type],
            self._arm_raised_holds,
        )

        # -- assembly predicate --------------------------------------------
        self._Assembled = Predicate(
            "assembled",
            [self._part_type, self._part_type],
            self._assembled_holds,
        )

        # Static objects
        self._robot = Object("robot", self._robot_type)
        self._table = Object("table", self._table_type)

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "furniture_bench_thread"

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type,
            self._part_type,
            self._thread_type,
            self._robot_type,
            self._table_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HasThread,
            self._Occupied,
            self._Aligned,
            self._Touching,
            self._Holding,
            self._AroundPart,
            self._HoveringAbovePart,
            self._HoveringAboveThread,
            self._GripperOpen,
            self._ArmRaised,
            self._Assembled,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Assembled}

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        return {self._Assembled}

    @property
    def action_space(self) -> Box:
        return Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 3.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError()

    def render_state_plt(self, state, task, action=None, caption=None):
        raise NotImplementedError()

    def render_state(self, state, task, action=None, caption=None):
        raise NotImplementedError()

    def get_event_to_action_fn(self) -> Callable:
        raise NotImplementedError()

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        return self._copy_observation(self._current_observation)

    def step(self, action: Action) -> Observation:
        self._current_observation = self.simulate(self._current_observation, action)
        return self._copy_observation(self._current_observation)

    def _generate_train_tasks(self):
        raise NotImplementedError()

    def _generate_test_tasks(self):
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Predicate helpers — read from simulator_state ground atoms
    # ------------------------------------------------------------------

    @classmethod
    def _pred_util(cls, state: State, pred_name: str, objects: Sequence[Object]) -> bool:
        """Look up a ground atom in state.simulator_state."""
        sim_state = state.simulator_state
        assert sim_state is not None
        given_objs = [o.name for o in objects]
        if len(set(given_objs)) != len(given_objs):
            return False
        for e in sim_state:
            if e.predicate.name.lower() != pred_name.lower():
                continue
            e_objs = [o.name for o in e.objects]
            if e_objs != given_objs:
                continue
            return not e.predicate.is_negated
        raise RuntimeError(
            f"Predicate {pred_name} with objects {given_objs} not found in simulator state."
        )

    @classmethod
    def _has_thread_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "has_thread", objects)

    @classmethod
    def _occupied_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "occupied", objects)

    @classmethod
    def _aligned_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "aligned", objects)

    @classmethod
    def _touching_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "touching", objects)

    @classmethod
    def _holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "holding", objects)

    @classmethod
    def _around_part_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "around_part", objects)

    @classmethod
    def _hovering_above_part_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "hovering_above_part", objects)

    @classmethod
    def _hovering_above_thread_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "hovering_above_thread", objects)

    @classmethod
    def _gripper_open_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "gripper_open", objects)

    @classmethod
    def _arm_raised_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "arm_raised", objects)

    @classmethod
    def _assembled_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._pred_util(state, "assembled", objects)
