import numpy as np
from PIL import Image
from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import (
    _Option,
    GroundAtom,
    Object,
    ParameterizedOption,
    State,
)


utils.update_config(
    {
        "seed": 0,
        # "pybullet_sim_steps_per_action": 100,
    }
)
env = create_new_env("pybullet_coffee", do_cache=True, use_gui=True)
env.reset(train_or_test="train", task_idx=0)


def _get_sampler_for_option(option: ParameterizedOption):
    if option.params_space.shape == (0,):
        return utils.null_sampler
    else:
        match option.name:
            case "TwistJug":
                return lambda _, __, rng, ___: np.array(
                    rng.uniform(-1, 1, size=(1,)), dtype=np.float32
                )
            case _:
                raise NotImplementedError(f"Unknown option {option.name}")


rng = np.random.default_rng(seed=0)


def _sample_option_from_nsrt(
    option: ParameterizedOption,
    objects: list[Object],
    state: State,
    atoms: set[GroundAtom],
    goal: set[GroundAtom],
) -> _Option:
    # copied from predicators/approaches/nsrt_metacontroller_approach.py:l48
    """Given a ground NSRT, try invoking its sampler repeatedly until we
    find an option that produces the expected next atoms under the ground
    NSRT."""
    metacontroller_max_samples = 100

    sampler = _get_sampler_for_option(option)
    # option_model = create_option_model("oracle")
    for _ in range(metacontroller_max_samples):
        # Invoke the ground NSRT's sampler to produce an option.
        params = sampler(state, set(), rng, [])
        opt = option.ground(objects, params)
        if not opt.initiable(state):
            # The option is not initiable. Continue on to the next sample.
            continue
        # try:
        #     next_state, _ = option_model.get_next_state_and_num_actions(state, opt)
        # except utils.EnvironmentFailure:
        #     continue
        # expected_next_atoms = utils.apply_operator(ground_nsrt, atoms)
        # if not all(a.holds(next_state) for a in expected_next_atoms):
        #     # Some expected atom is not achieved. Continue on to the
        #     # next sample.
        #     continue
        return opt
    raise RuntimeError("Failed to execute the motion in the current state.")


def run_motion(motion: str):
    motion_name, motion_args = motion.split("(")
    motion_args = motion_args[:-1]  # remove trailing ")"
    motion_args = [arg.strip()[1:-1] for arg in motion_args.split(",")]

    options = get_gt_options(env.get_name())
    option = next((opt for opt in options if opt.name == motion_name))

    state = env._current_state
    all_objs = list(state.data.keys())
    all_objs = {obj.name: obj for obj in all_objs}

    objects = [all_objs[arg] for arg in motion_args]

    # run motion
    cur_option = _sample_option_from_nsrt(option, objects, state, set(), set())
    for _ in range(100):
        act = cur_option.policy(state)
        state = env.step(act)
        if cur_option.terminal(state):
            break
    assert cur_option.terminal(state), "Failed to execute the motion in the current state."


def _render():
    images = env.render()
    image = Image.fromarray(images[0])
    image.save("image2.png")


def main():
    _render()
    run_motion("MoveToTwistJug('robby', 'juggy')")
    _render()
    run_motion("RotateItemUntilHandleAccessible('robby', 'juggy')")
    _render()
    run_motion("PickJug('robby', 'juggy')")
    _render()


if __name__ == "__main__":
    main()
