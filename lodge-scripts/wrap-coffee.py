from pathlib import Path
from typing import Generator

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.base_env import BaseEnv
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


def main():
    args = utils.parse_args()
    utils.update_config(args)
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    domain_name = env.get_name()

    exp_dir = Path("/home/claudius-kienle/Documents/Projects/DEA/digital-engineering-assistant/experiments/industrycap-paper/demos/coffee/data/problems")

    for i, task in enumerate(parse_tasks(env, domain_name)):
        (exp_dir / f"p{i:02d}.pddl").write_text(task)


if __name__ == "__main__":
    main()
