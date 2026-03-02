from pddl_utils import Object, Type, Predicate, GroundAtom
import json
from sympy import re
from pathlib import Path
from typing import Generator

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import EnvironmentTask


def to_snake(pascal: str) -> str:
    """Converts a Pascal case string to snake case."""
    magic = re.findall("[A-Z]+[a-z]*", pascal)
    snake = "_".join(magic)
    snake = snake.lower()

    return snake


def function_mapping(nsrts: set):
    function_stubs = {}
    for nsrt in nsrts:
        op_vars = nsrt.op.parameters

        option_name = nsrt.option.name
        option_vars = nsrt.option_vars
        assert option_vars[0].type.name == "robot"  # in this env, we always also pass the robot
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


def task_to_pddl_problem(task: EnvironmentTask, env: BaseEnv, domain_name: str, problem_name: str) -> str:
    # Extract objects from the initial state
    objects = set(task.init)

    init_atoms = utils.abstract(task.init, env.predicates)
    goal = task.goal

    # Use the existing utility function to create the PDDL problem string
    return utils.create_pddl_problem(objects, init_atoms, goal, domain_name, problem_name)


def parse_tasks(env: BaseEnv, domain_name: str) -> Generator[str, None, None]:
    tasks = env.get_test_tasks()

    # Example usage: convert first task to PDDL
    for i, task in enumerate(tasks):
        pddl_problem = task_to_pddl_problem(task, env, domain_name, f"task_{i}")
        yield pddl_problem


LODGE_EXP_DIR = Path(__file__).parent.parent.parent.parent


def gen_data():
    utils.update_config(
        {
            "seed": 0,
            "pybullet_robot": "panda",
            # "pybullet_sim_steps_per_action": 100,
            "pybullet_camera_width": 640,
            "pybullet_camera_height": 480,
        }
    )
    env = create_new_env("pybullet_coffee", do_cache=True, use_gui=False)
    domain_name = env.get_name()

    exp_dir = LODGE_EXP_DIR / "demos/predicators/data" / env.get_name()
    assert exp_dir.is_dir(), f"Expected {exp_dir} to be a directory"

    if True:
        from pddl_utils import parse_problem

        # generate problems
        problems = set()
        for i, task in enumerate(parse_tasks(env, domain_name)):
            p = parse_problem(task, domain=None)
            problems.add((p.objects, p.init, p.goal))
        example_problem = parse_problem(parse_tasks(env, domain_name).__next__(), domain=None)
        # (exp_dir / "problems" / f"p{i:02d}.pddl").write_text(task)
        add_inits = [frozenset({GroundAtom(Predicate("JugPickable", types=[Type("jug")]), [Object("juggy", Type("jug"))])})]
        i = 0
        for objs, init, goal in problems:
            pddl_problem = example_problem.copy_with(objects=objs, init=init, goal=goal)
            (exp_dir / "problems" / f"p{i:02d}.pddl").write_text(pddl_problem.to_string())
            i += 1
            for add_init in add_inits:
                alt_problem = pddl_problem.copy_with(init=add_init | init, goal=goal)
                (exp_dir / "problems" / f"p{i:02d}.pddl").write_text(alt_problem.to_string())
                i += 1
    return
    if True:
        # generate domain
        preds = env.predicates
        options = get_gt_options(env.get_name())

        nsrts = get_gt_nsrts(env.get_name(), predicates_to_keep=preds, options_to_keep=options)

        domain_pddl = utils.create_pddl_domain(nsrts, preds, env.types, domain_name)
        (exp_dir / "domain.pddl").write_text(domain_pddl)

        # domain skeleton
        domain_pddl = utils.create_pddl_domain([], env.goal_predicates, env.types, domain_name)
        (exp_dir / "domain_skeleton.pddl").write_text(domain_pddl)

        (exp_dir / "function_mapping.json").write_text(json.dumps(function_mapping(nsrts), indent=4))

    if False:
        # generate function stubs (options)
        options = get_gt_options(env.get_name())
        _MOTION_NAME_ALIASES_INV = {v: k for k, v in _MOTION_NAME_ALIASES.items()}
        func_subs_str = ""
        for option in options:
            option_name = _MOTION_NAME_ALIASES_INV.get(option.name, option.name)
            params = ", ".join([f"p{i}: {var.name.capitalize()}" for i, var in enumerate(option.types)])
            func_subs_str += f"def {option_name}({params}): ...\n\n"
        (exp_dir / "function_stubs.py").write_text(func_subs_str)

    if False:
        # generate domain knowledge
        (exp_dir / "domain_knowledge.txt").write_text(env.__doc__ or "")


if __name__ == "__main__":
    gen_data()
