import json
import os
import subprocess
import tempfile

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

from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import (
    _Option,
    GroundAtom,
    Object,
    ParameterizedOption,
    State,
)

assert os.environ.get("PYTHONHASHSEED") == "0"


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


_BOOL_FEATURES = {"is_on", "is_filled", "is_held", "handle_accessible"}


def remap_state_to_dict(state: State, env_name: str | None = None, raw: bool = False) -> dict[str, dict[str, float | bool]]:
    """Remap state features into a dict for each object."""
    result = {}
    for obj in state.data:
        feature_dict: dict[str, float | bool] = {}
        for i, feature_name in enumerate(obj.type.feature_names):
            feature_dict[feature_name] = state.data[obj][i]
        
        if not raw:
            if env_name == "pybullet_coffee":
                if obj.type.name == "robot":
                    feature_dict.pop("tilt", None)
                    feature_dict.pop("wrist", None)
                    feature_dict.pop("fingers", None)
                if obj.type.name == "jug" and "rot" in feature_dict:
                    rot = float(feature_dict.pop("rot"))
                    feature_dict["handle_accessible"] = abs(abs(rot) - np.pi) < 0.75
                if obj.type.name == "machine":
                    feature_dict['x'] -= 0.1
            else:
                raise NotImplementedError(f"Unknown environment name {env_name}")

            # Cast float flags (0.0 / 1.0) to proper booleans.
            for key in _BOOL_FEATURES:
                if key in feature_dict:
                    feature_dict[key] = bool(feature_dict[key] > 0.5)

        result[f"{obj.name}:{obj.type.name}"] = feature_dict
    return result


app = FastAPI()

# args = utils.parse_args()
utils.update_config(
    {
        "seed": 0,
        "pybullet_robot": "panda",
        # "pybullet_sim_steps_per_action": 100,
        "pybullet_camera_width": 640,
        "pybullet_camera_height": 480,
    }
)
env: PyBulletCoffeeEnv = create_new_env("pybullet_coffee", do_cache=True, use_gui=False)  # type: ignore

states = StateStorage()

_recording_frames: list[Image.Image] = []
_is_recording: bool = False


def _get_curr_state() -> State:
    return env._current_observation


def _get_image() -> Image.Image:
    try:
        image = env.render_plt()
        image = image.get_figure()
        assert image is not None
    except (NotImplementedError, AssertionError):
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
    assert env.get_name() == env_name, f"Expected environment name {env_name} but got {env.get_name()}"


@app.post("/start-recording", operation_id="start_recording")
def start_recording():
    global _recording_frames, _is_recording
    _recording_frames = []
    _is_recording = True
    # Capture initial frame
    _recording_frames.append(_get_image())
    print("Recording started")


@app.get("/stop-recording", operation_id="stop_recording")
def stop_recording(output_path: str = "recording.mp4") -> str:
    global _recording_frames, _is_recording
    _is_recording = False
    if not _recording_frames:
        return "No frames recorded."

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(_recording_frames):
            frame.save(os.path.join(tmpdir, f"frame_{i:05d}.png"))

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", "10",
                "-i", os.path.join(tmpdir, "frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    _recording_frames = []
    print(f"Recording saved to {output_path}")
    return output_path


@app.post("/reset", operation_id="reset")
def reset(task_idx: int):
    # env.restart_recording()
    env.reset(train_or_test="train", task_idx=task_idx)

    image = _get_image()
    image.save("image2.png")
    if _is_recording:
        _recording_frames.append(image)
    _add_curr_state()


@app.post("/get-env-hash", operation_id="get_env_hash")
def get_env_hash() -> str:
    return _get_curr_hash()


@app.post("/set-env-hash", operation_id="set_env_hash")
def set_env_hash(s_hash: str):
    s = states[s_hash]
    assert s is not None, f"State with hash {s_hash} not found in storage."
    env._current_observation = s
    env._reset_state(s)

    _get_image().save("image2.png")

    assert _get_curr_hash() == s_hash


@app.get("/state", operation_id="get_state")
def get_state(raw: bool = False) -> dict[str, dict[str, float | bool]]:
    state = env.get_observation()
    return remap_state_to_dict(state, env.get_name(), raw=raw)


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
                return lambda _, __, rng, ___: np.array(rng.uniform(-1, 1, size=(1,)), dtype=np.float32)
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

    init_state = _get_curr_state()
    state = init_state.copy()
    all_objs = list(state.data.keys())
    all_objs = {obj.name: obj for obj in all_objs}

    objects = [all_objs[arg] for arg in motion_args]

    try:
        # run motion
        cur_option = _sample_option_from_nsrt(option, objects, state, set(), set())
        for _ in range(100):
            act = cur_option.policy(state)
            state = env.step(act)
            if _is_recording:
                _recording_frames.append(_get_image())
            if cur_option.terminal(state):
                break
        if not cur_option.terminal(state):
            # raise RuntimeError("Failed to execute the motion in the current state.")
            print("Warning: Option did not terminate within 100 steps. Forcing termination.")
            env._current_observation = init_state
            env._reset_state(init_state)

    except RuntimeError as e:
        # TODO: reset environment
        return RunMotionResponseModel(error_response=str(e))
    except utils.OptionExecutionFailure as e:
        return RunMotionResponseModel(error_response="The motion failed to execute successfully.")
    except TypeError as e:
        return RunMotionResponseModel(error_response="The motion command is invalid.", translation=True)

    _add_curr_state()
    post_state = _get_curr_hash()

    print("State changed: %s -> %s" % (prev_state, post_state))
    _get_image().save("image2.png")

    return RunMotionResponseModel(error_response=None)


class GetPredicatesEvaluationRequestModel(BaseModel):
    all_predicates: bool = False


@app.post("/predicates-evaluation", operation_id="get_predicates_evaluation")
def get_predicates_evaluation(all_predicates: bool = False) -> list[str]:
    state = env.get_observation()
    predicates = env.goal_predicates if not all_predicates else env.predicates
    all_atoms = utils.all_possible_ground_atoms(state, predicates)
    evaluated_literals: list[str] = []
    for atom in all_atoms:
        atom_str = atom.pddl_str()
        if atom.holds(state):
            evaluated_literals.append(atom_str)
        else:
            evaluated_literals.append(f"(not {atom_str})")
    return evaluated_literals


@app.get("/get-supported-predicates", operation_id="get_supported_predicates")
def get_supported_predicates() -> list[str]:
    return [p.name for p in env.goal_predicates]
