"""A simple gridworld environment where a robot prepares a burger, inspired by
https://github.com/portal-cornell/robotouille.

This environment uses assets from robotouille that were designed by
Nicole Thean (https://github.com/nicolethean).
"""

import copy
import io
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from PIL import Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, DefaultEnvironmentTask, \
    EnvironmentTask, GroundAtom, Object, Observation, Predicate, State, Task, \
    Type, Video


class FurnitureBenchEnv(BaseEnv):
    """A simple gridworld environment where a robot prepares a burger, inspired
    by https://github.com/portal-cornell/robotouille.

    This environment is designed to showcase a predicate invention approach that
    learns geometric predicates that operate on the object-centric state and
    vision-language model predicates that operate on the visual rendering of the
    state.

    One quirk of this environment is that we want certain parts of the state to
    only be accessible by the oracle approach. This is because we want to invent
    predicates like IsCooked as a VLM predicate, but not a geometric predicate,
    so no information about how cooked an object is should be in the state. The
    solution to this is to hide certain state inside State.simulator_state.
    After the demonstrations are created by the oracle approach, we can erase
    this simulator state before we pass the demonstrations to the predicate
    invention approach.

    Example command to see demos created:
    python predicators/main.py --env burger
    --approach grammar_search_invention --seed 0 --num_train_tasks 10
    --option_model_terminate_on_repeat False
    --sesame_max_skeletons_optimized 1000 --timeout 80
    --make_demo_videos
    --bilevel_plan_without_sim True
    --sesame_task_planner fdopt

    Note that the default task planner is too slow -- fast downward is required.
    """

    # Types
    _object_type = Type("object", [])
    _part_type = Type("part", [], _object_type)
    _table_type = Type("table", [], _object_type)
    _robot_type = Type("robot", [], _object_type)

    dir_to_enum = {"up": 0, "left": 1, "down": 2, "right": 3}
    enum_to_dir = {0: "up", 1: "left", 2: "down", 3: "right"}

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._Aligned = Predicate("Aligned",
                                   [self._part_type, self._part_type],
                                   self.Aligned)
        self._Assembled = Predicate("Assembled",
                                    [self._part_type, self._part_type],
                                    self.Assembled)
        self._GripperAround = Predicate("GripperAround",
                                        [self._part_type],
                                        self._GripperAround_holds)
        self._Holding = Predicate("Holding",
                                  [self._part_type],
                                  self._Holding_holds)
        self._HoveredAbove = Predicate("HoveredAbove",
                                       [self._part_type],
                                       self._HoveredAbove_holds)
        self._OnTable = Predicate("OnTable",
                                  [self._part_type],
                                  self._OnTable_holds)
        self._Touching = Predicate("Touching",
                                   [self._part_type, self._part_type],
                                   self._Touching_holds)
        self._Screwable = Predicate("Screwable",
                                    [self._part_type, self._part_type],
                                    self._Screwable_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robot", self._robot_type)
        self._table = Object("table", self._table_type)

    @classmethod
    def get_name(cls) -> str:
        return "furniture_bench"

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._part_type, self._table_type, self._robot_type
        }

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               rng=self._train_rng,
                               train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng,
                               train_or_test="test")

    @classmethod
    def pred_util(cls, state: State, pred_name: str, objects: Sequence[Object]) -> bool:
        """Public for use by oracle options."""
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
        raise RuntimeError(f"Predicate {pred_name} with objects {given_objs} not found in simulator state.")

    @classmethod
    def Aligned(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls.pred_util(state, "aligned", objects)

    @classmethod
    def Assembled(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls.pred_util(state, "assembled", objects)

    def _GripperAround_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is positioned around a part."""
        return self.pred_util(state, "gripper_around", objects)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if a part is being held by the gripper."""
        return self.pred_util(state, "holding", objects)

    def _HoveredAbove_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is hovered above a part."""
        return self.pred_util(state, "hovered_above", objects)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if a part is on the table."""
        return self.pred_util(state, "on_table", objects)

    def _Touching_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if two parts are touching each other."""
        return self.pred_util(state, "touching", objects)

    def _Screwable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if two parts can be screwed together."""
        return self.pred_util(state, "screwable", objects)


    @classmethod
    def get_position(cls, obj: Object, state: State) -> Tuple[int, int]:
        """Public for use by oracle options."""
        col = state.get(obj, "col")
        row = state.get(obj, "row")
        return col, row

    @classmethod
    def is_adjacent(cls, col_1: int, row_1: int, col_2: int,
                    row_2: int) -> bool:
        """Public for use by oracle options."""
        adjacent_vertical = col_1 == col_2 and abs(row_1 - row_2) == 1
        adjacent_horizontal = row_1 == row_2 and abs(col_1 - col_2) == 1
        return adjacent_vertical or adjacent_horizontal

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._GripperAround, self._Holding, self._HoveredAbove,
            self._OnTable, self._Touching, self._Screwable, self._Aligned,
            self._Assembled
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Assembled}

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        return {self._Assembled}

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), direction, cut/cook, pick/place
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect direction to be one of -1, 0, 1, 2, or 3. -1 signifies
        # "no change in direction", and 0, 1, 2, and 3 signify a direction.
        # We expect cut/cook and pick/place to be 0 or 1.
        return Box(low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
                   high=np.array([1.0, 1.0, 3.0, 1.0, 1.0]),
                   dtype=np.float32)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError()

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError()

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError()

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        # Rather than have the observation be the state + the image, we just
        # package the image inside the state's simulator_state.
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        return self._copy_observation(self._current_observation)

    def step(self, action: Action) -> Observation:
        # Rather than have the observation be the state + the image, we just
        # package the image inside the state's simulator_state.
        self._current_observation = self.simulate(self._current_observation,
                                                  action)
        return self._copy_observation(self._current_observation)

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        raise NotImplementedError()