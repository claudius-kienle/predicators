import json
from pathlib import Path
import logging
import pickle
import re
import time

from PIL import Image
import numpy as np
from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan
from predicators.datasets.generate_atom_trajs_with_vlm import (
    _generate_ground_atoms_with_vlm_pure_visual_preds,
)
from predicators.envs import create_new_env
from predicators.envs.base_env import BaseEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _run_pipeline
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import (
    NSRT,
    Action,
    Dataset,
    ImageOptionTrajectory,
    LiftedAtom,
    LowLevelTrajectory,
    ParameterizedOption,
    State,
    Task,
    Variable,
)


lodge_handover = Path(__file__).parent.parent / "lodge-handover"


def to_snake(pascal: str) -> str:
    """Converts a Pascal case string to snake case."""
    magic = re.findall("[A-Z]+[a-z]*", pascal)
    snake = "_".join(magic)
    snake = snake.lower()

    return snake


def create_dataset(env: BaseEnv, options: set[ParameterizedOption]):
    """Create offline datasets for training, given a set of training tasks for
    an environment.

    Some or all of this data may be loaded from disk.
    """
    from predicators.structs import _Option, Object, Type

    data = pickle.load(open(lodge_handover / "demos.pkl", "rb"))
    demos = data["demos"]
    objs = data["objects"]
    goal = data["goal"]

    object_type = Type("object", feature_names=[])
    objs = {
        obj["name"]: Object(
            name=obj["name"],
            type=Type(obj["type"], feature_names=[], parent=object_type),
        )
        for obj in objs
    }

    goal_p = []
    for goal_pred in goal.split("(and")[1][:-1].strip().split("\n\t"):
        goal_content = goal_pred[1:-1].split()
        p_name = goal_content[0]
        p_args = goal_content[1:]
        pred = next(iter([p for p in env.predicates if p.name.lower() == p_name]))
        p_objs = {
            Variable(name=f"?{i}", type=objs[p_arg].type): objs[p_arg]
            for i, p_arg in enumerate(p_args)
        }
        lift_atom = LiftedAtom(pred, tuple(p_objs.keys()))
        goal_p.append(lift_atom.ground(p_objs))

    all_objs = set(objs.values())

    img_option_trajs: list[ImageOptionTrajectory] = []
    option_segmented_trajs: list[LowLevelTrajectory] = []
    train_tasks: list[Task] = [
        Task(
            init=State({o: np.array([]) for o in all_objs}),
            goal=set(goal_p),
        )
        for _ in range(len(demos))
    ]
    demo_lengths = [len(d["states"]) for d in demos]
    for idx, demo in enumerate(demos):
        state_img_paths = demo["images"]
        state_actions = demo["actions"]
        states = demo["states"]

        state_imgs = [
            [Image.open(state_img_path)] for state_img_path in state_img_paths
        ]
        cropped_state_imgs = []

        curr_traj_states_for_vlm: list[State] = []
        for s in states:
            curr_traj_states_for_vlm.append(
                State({obj: np.array([]) for obj in all_objs}, simulator_state=s)
            )

        curr_traj_actions_for_vlm: list[Action] = []
        for state_action in state_actions:
            action_name, rest = state_action.split("(", 1)
            # Convert snake_case to CamelCase
            action_name = "".join(word.capitalize() for word in action_name.split("_"))
            action_args = [
                arg.strip()[1:-1] for arg in rest[:-1].split(",") if arg.strip()
            ]  # Remove trailing ')'

            op = next(filter(lambda o: o.name == action_name, options))
            args = [objs[arg] for arg in action_args]
            args.insert(0, objs["arm"])  # in this env, we always also pass the robot
            g_op = op.ground(args, np.zeros(0, dtype=float))
            curr_traj_actions_for_vlm.append(Action(np.zeros(0, dtype=float), g_op))

        is_demo = len(states) == max(demo_lengths)
        img_option_trajs.append(
            ImageOptionTrajectory(
                all_objs,
                state_imgs,
                cropped_state_imgs,
                [act.get_option() for act in curr_traj_actions_for_vlm],
                curr_traj_states_for_vlm,
                is_demo,
                idx,
            )
        )
        option_segmented_trajs.append(
            LowLevelTrajectory(
                curr_traj_states_for_vlm,
                [
                    Action(np.zeros(act.arr.shape, dtype=float), act.get_option())
                    for act in curr_traj_actions_for_vlm
                ],
                is_demo,
                idx,
            )
        )

    known_predicates = env.goal_predicates
    vlm = utils.create_vlm_by_name("gpt-4.1-mini")  # pragma: no cover

    # mimics predicators/datasets/generate_atom_trajs_with_vlm.py:1116
    ground_atoms_traj = _generate_ground_atoms_with_vlm_pure_visual_preds(
        img_option_trajs, env, train_tasks, known_predicates, all_objs, vlm
    )

    return Dataset(option_segmented_trajs, ground_atoms_traj), train_tasks


def function_mapping(nsrts: list[NSRT]):
    function_stubs = {}
    for nsrt in nsrts:
        op_vars = nsrt.op.parameters

        option_name = nsrt.option.name
        option_vars = nsrt.option_vars
        assert (
            option_vars[0].type.name == "robot"
        )  # in this env, we always also pass the robot
        option_vars = op_vars[1:]

        arg_mapping: list[int | None] = []
        for op_var in op_vars:
            if op_var in option_vars:
                arg_mapping.append(option_vars.index(op_var))
            else:
                arg_mapping.append(None)

        function_stubs[nsrt.op.name] = {
            "name": to_snake(option_name),
            "arg_mapping": arg_mapping,
        }
    return function_stubs


def main():
    args = utils.parse_args()
    utils.update_config(args)

    script_start = time.perf_counter()

    logging.basicConfig(level=logging.INFO, force=True)

    env = create_new_env("furniture_bench", do_cache=True, use_gui=False)
    perceiver = create_perceiver(CFG.perceiver)

    preds = env.goal_predicates

    # If we are not doing option learning, pass in all the environment's
    # oracle options.
    options = get_gt_options(env.get_name())

    offline_dataset, train_tasks = create_dataset(env, options)

    approach_train_tasks = train_tasks

    # Create the agent (approach).
    approach_name = CFG.approach
    if CFG.approach_wrapper:
        approach_name = f"{CFG.approach_wrapper}[{approach_name}]"
    approach = create_approach(
        approach_name, preds, options, env.types, env.action_space, approach_train_tasks
    )

    # Create the cognitive manager.
    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, perceiver, execution_monitor)
    # Run the full pipeline.
    _run_pipeline(env, cogman, approach_train_tasks, offline_dataset)
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")

    nsrts: list[NSRT] = approach._get_current_nsrts()
    predicates = approach._get_current_predicates()

    pddl_ops = [nsrt.pddl_str() for nsrt in nsrts]
    pddl_preds = [pred.pddl_str() for pred in predicates]
    pddl_types = " ".join(
        [
            f"{t.name} - {t.parent.name}" if t.parent is not None else t.name
            for t in env.types
        ]
    )

    pddl_domain = f"""(define (domain furniture_bench)
    (:requirements :strips :typing :universal-preconditions :negative-preconditions :disjunctive-preconditions :existential-preconditions :conditional-effects :equality)
    (:types {pddl_types})
    (:predicates
        {' '.join(pddl_preds)}
    )
    {' '.join(pddl_ops)}
)"""

    (lodge_handover / "proposed_domain.pddl").write_text(pddl_domain)
    (lodge_handover / "function_stubs.json").write_text(
        json.dumps(function_mapping(nsrts), indent=4)
    )


if __name__ == "__main__":
    main()
