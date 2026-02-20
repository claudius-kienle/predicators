from pathlib import Path
from typing import Generator

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import EnvironmentTask


def task_to_pddl_problem(
    task: EnvironmentTask, env: BaseEnv, domain_name: str, problem_name: str
) -> str:
    # Extract objects from the initial state
    objects = set(task.init)

    init_atoms = utils.abstract(task.init, env.predicates)
    goal = task.goal

    # Use the existing utility function to create the PDDL problem string
    return utils.create_pddl_problem(
        objects, init_atoms, goal, domain_name, problem_name
    )


def parse_tasks(env: BaseEnv, domain_name: str) -> Generator[str, None, None]:
    tasks = env.get_test_tasks()

    # Example usage: convert first task to PDDL
    for i, task in enumerate(tasks):
        pddl_problem = task_to_pddl_problem(task, env, domain_name, f"task_{i}")
        yield pddl_problem


LODGE_EXP_DIR = Path(__file__).parent.parent.parent.parent

def gen_data():
    utils.update_config({"seed": 0})
    env = create_new_env("pybullet_coffee", do_cache=True, use_gui=False)
    domain_name = env.get_name()

    exp_dir = LODGE_EXP_DIR / "demos/predicators/data" / env.get_name()
    assert exp_dir.is_dir(), f"Expected {exp_dir} to be a directory"

    if True:
        # generate problems
        for i, task in enumerate(parse_tasks(env, domain_name)):
            (exp_dir / "problems" / f"p{i:02d}.pddl").write_text(task)

    if False:
        # generate domain
        preds = env.predicates
        options = get_gt_options(env.get_name())

        nsrts = get_gt_nsrts(env.get_name(), predicates_to_keep=preds, options_to_keep=options)

        domain_pddl = utils.create_pddl_domain(nsrts, preds, env.types, domain_name)
        (exp_dir / "domain.pddl").write_text(domain_pddl)

        # domain skeleton
        domain_pddl = utils.create_pddl_domain([], env.goal_predicates, env.types, domain_name)
        (exp_dir / "domain_skeleton.pddl").write_text(domain_pddl)

    if True:
        # generate function stubs (options)
        options = get_gt_options(env.get_name())

        func_subs_str = ""
        for option in options:
            option_name = option.name
            params = ", ".join([f"p{i}: {var.name.capitalize()}" for i, var in enumerate(option.types)])
            func_subs_str += f"def {option_name}({params}): ...\n\n"
        (exp_dir / "function_stubs.py").write_text(func_subs_str)

    if True:
        # generate domain knowledge
        (exp_dir / "domain_knowledge.txt").write_text(env.__doc__ or "")


if __name__ == "__main__":
    gen_data()
