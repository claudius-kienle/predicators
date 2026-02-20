import json

import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI
from predicators import utils
from predicators.envs import create_new_env

import base64
import io
import pickle
from pathlib import Path

from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import create_option_model
from predicators.structs import (
    _Option,
    Array,
    GroundAtom,
    Object,
    ParameterizedOption,
    State,
)


class StateStorage:
    """Persistent storage for environment states."""

    def __init__(self, filepath: str = "states.pkl"):
        self.filepath = Path(filepath)
        self.states = self._load()

    def _load(self) -> dict:
        """Load states from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading states from {self.filepath}: {e}")
                return {}
        return {}

    def _save(self):
        """Save states to disk."""
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(self.states, f)
        except Exception as e:
            print(f"Error saving states to {self.filepath}: {e}")

    def __getitem__(self, key):
        return self.states[key]

    def __setitem__(self, key, value):
        self.states[key] = value
        self._save()

    def __contains__(self, key):
        return key in self.states


def image_to_base64(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


def remap_state_to_dict(state: State) -> dict[str, dict[str, float]]:
    """Remap state features into a dict for each object."""
    result = {}
    for obj in state.data:
        feature_dict = {}
        for i, feature_name in enumerate(obj.type.feature_names):
            feature_dict[feature_name] = state.data[obj][i]
        result[f"{obj.name}:{obj.type.name}"] = feature_dict
    return result


app = FastAPI()
tolerance = 5e-3

# args = utils.parse_args()
utils.update_config(
    {
        "seed": 0,
    }
)
env = create_new_env("pybullet_coffee", do_cache=True, use_gui=False)

states = StateStorage()


def _get_curr_state() -> State:
    return env._current_observation

def _get_image() -> Image.Image:
    try:
        image = env.render_plt()
        image = image.get_figure()
    except NotImplementedError:
        images = env.render()
        assert len(images) == 1, "Expected exactly one camera image"
        image = Image.fromarray(images[0])
    return image


def _get_curr_hash():
    s = _get_curr_state()
    s_data = {hash(k): [round(val, 6) for val in v.tolist()] for k, v in s.data.items()}
    return str(hash(json.dumps(s_data, sort_keys=True)))


def _add_curr_state():
    states[_get_curr_hash()] = _get_curr_state()


@app.post("/set-environment", operation_id="set_environment")
def set_environment(env_name: str):
    assert (
        env.get_name() == env_name
    ), f"Expected environment name {env_name} but got {env.get_name()}"


@app.get("/stop-recording", operation_id="stop_recording")
def close():
    print("stop-recording")
    raise NotImplementedError()


@app.post("/reset", operation_id="reset")
def reset(task_idx: int):
    # env.restart_recording()
    env.reset(train_or_test="train", task_idx=task_idx)

    _get_image().save("image2.png")
    _add_curr_state()


@app.post("/get-env-hash", operation_id="get_env_hash")
def get_env_hash() -> str:
    return _get_curr_hash()


@app.post("/set-env-hash", operation_id="set_env_hash")
def set_env_hash(s_hash: str):
    s = states[s_hash]
    env._current_observation = s

    assert _get_curr_hash() == s_hash


@app.get("/state", operation_id="get_state")
def get_state() -> dict[str, dict[str, float]]:
    state = env.get_observation()
    return remap_state_to_dict(state)


@app.get("/image", operation_id="get_image")
def get_image() -> str:
    """Get the current camera observation as a base64 encoded image.

    Returns:
        dict: Dictionary containing base64 encoded PNG image
    """
    # image = get_image_with_labels(env)
    image = _get_image()

    return image_to_base64(image)


class RunMotionResponseModel(BaseModel):
    error_response: str | None
    translation: bool = False


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


@app.post("/run-motion", operation_id="run_motion")
def run_motion(motion: str) -> RunMotionResponseModel:
    print(motion)

    prev_state = _get_curr_hash()

    motion_name, motion_args = motion.split("(")
    motion_args = motion_args[:-1]  # remove trailing ")"
    motion_args = [arg.strip()[1:-1] for arg in motion_args.split(",")]

    options = get_gt_options(env.get_name())
    option = next((opt for opt in options if opt.name == motion_name))

    state = _get_curr_state()
    all_objs = list(state.data.keys())
    all_objs = {obj.name: obj for obj in all_objs}

    objects = [all_objs[arg] for arg in motion_args]

    try:
        # run motion
        cur_option = _sample_option_from_nsrt(option, objects, state, set(), set())
        act = cur_option.policy(state)
        env.step(act)

    except RuntimeError as e:
        # TODO: reset environment
        return RunMotionResponseModel(error_response=str(e))
    except TypeError as e:
        return RunMotionResponseModel(
            error_response="The motion command is invalid.", translation=True
        )

    _add_curr_state()
    post_state = _get_curr_hash()

    print("State changed: %s -> %s" % (prev_state, post_state))

    return RunMotionResponseModel(error_response=None)


@app.post("/predicates-evaluation", operation_id="get_predicates_evaluation")
def get_predicates_evaluation() -> list[str]:
    atoms = utils.abstract(env.get_observation(), env.goal_predicates)
    return [atom.pddl_str() for atom in atoms]


@app.get("/get-supported-predicates", operation_id="get_supported_predicates")
def get_supported_predicates() -> list[str]:
    return [p.name for p in env.goal_predicates]
