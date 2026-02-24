"""Generic controllers for the robots."""

from typing import Callable, Dict, Optional, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box
from pybullet_utils.transformations import euler_from_quaternion

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import InverseKinematicsError
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, State, Type

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda"}


def create_move_end_effector_to_pose_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]
    ],
    move_to_pose_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status."""

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " + f"not implemented for robot {robot_name}."
    )

    def _policy(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> Action:
        del memory  # unused
        # Sync the joints.
        assert isinstance(state, utils.PyBulletState)
        robot.set_joints(state.joint_positions)
        # First handle the main arm joints.
        current_pose, target_pose, finger_status = (
            get_current_and_target_pose_and_finger_status(state, objects, params)
        )
        # This option currently assumes a fixed end effector orientation.
        assert np.allclose(current_pose.orientation, target_pose.orientation)
        orn = current_pose.orientation
        current = current_pose.position
        target = target_pose.position
        # Run IK to determine the target joint positions.
        ee_delta = np.subtract(target, current)
        # Reduce the target to conform to the max velocity constraint.
        ee_norm = np.linalg.norm(ee_delta)
        if ee_norm > max_vel_norm:
            ee_delta = ee_delta * max_vel_norm / ee_norm
        dx, dy, dz = np.add(current, ee_delta)
        ee_action = Pose((dx, dy, dz), orn)
        # Keep validate as False because validate=True would update the
        # state of the robot during simulation, which overrides physics.
        try:
            # For the panda, always set the joints after running IK because
            # IKFast is very sensitive to initialization, and it's easier to
            # find good solutions on subsequent calls if we are already near
            # a solution from the previous call. The fetch robot does not
            # use IKFast, and in fact gets screwed up if we set joints here.
            joint_positions = robot.inverse_kinematics(
                ee_action, validate=False, set_joints=True
            )
        except InverseKinematicsError:
            raise utils.OptionExecutionFailure("Inverse kinematics failed.")
        # Handle the fingers. Fingers drift if left alone.
        # When the fingers are not explicitly being opened or closed, we
        # nudge the fingers toward being open or closed according to the
        # finger status.
        if finger_status == "open":
            finger_delta = finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -finger_action_nudge_magnitude
        # Extract the current finger state.
        state = cast(utils.PyBulletState, state)
        finger_position = state.joint_positions[robot.left_finger_joint_idx]
        # The finger action is an absolute joint position for the fingers.
        f_action = finger_position + finger_delta
        # Override the meaningless finger values in joint_action.
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        # This clipping is needed sometimes for the joint limits.
        action_arr = np.clip(
            action_arr, robot.action_space.low, robot.action_space.high
        )
        assert robot.action_space.contains(action_arr)
        return Action(action_arr)

    def _terminal(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = get_current_and_target_pose_and_finger_status(
            state, objects, params
        )
        # This option currently assumes a fixed end effector orientation.
        assert np.allclose(current_pose.orientation, target_pose.orientation)
        current = current_pose.position
        target = target_pose.position
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        return squared_dist < move_to_pose_tol

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=_terminal,
    )


def create_move_end_effector_to_pose_and_orientation_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]
    ],
    move_to_pose_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
    extra_terminal: Optional[
        Callable[[State, Sequence[Object], Array], bool]
    ] = None,
) -> ParameterizedOption:
    """Like create_move_end_effector_to_pose_option, but WITHOUT the fixed-
    orientation restriction.  Both position *and* orientation are moved toward
    the target each step.  A simple slerp-style step is used for orientation:
    the quaternion delta is clamped so its rotation angle does not exceed
    ``max_vel_norm`` radians per step, keeping motion smooth.

    Use this instead of ``create_move_end_effector_to_pose_option`` whenever
    the target orientation differs from the current one (e.g. tilting)."""

    from pybullet_utils.transformations import (
        quaternion_multiply,
        quaternion_conjugate,
        quaternion_slerp,
    )

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose+orientation option "
        f"not implemented for robot {robot_name}."
    )

    def _step_toward_pose(current_pose: Pose, target_pose: Pose) -> Pose:
        """Return a Pose one step closer to target (position + orientation)."""
        # --- position step ---
        current = np.array(current_pose.position)
        target_pos = np.array(target_pose.position)
        ee_delta = target_pos - current
        ee_norm = np.linalg.norm(ee_delta)
        if ee_norm > max_vel_norm:
            ee_delta = ee_delta * max_vel_norm / ee_norm
        next_pos = tuple(current + ee_delta)

        # --- orientation step (bounded slerp) ---
        curr_orn = np.array(current_pose.orientation)  # (x,y,z,w)
        tgt_orn = np.array(target_pose.orientation)
        # Clamp t so the angular step ≤ max_vel_norm radians.
        dot = float(np.clip(np.dot(curr_orn, tgt_orn), -1.0, 1.0))
        if dot < 0:  # take the shorter arc
            tgt_orn = -tgt_orn
            dot = -dot
        angle = 2.0 * np.arccos(abs(dot))
        if angle < 1e-9:
            next_orn = curr_orn
        else:
            t = min(1.0, max_vel_norm / angle)
            next_orn = quaternion_slerp(curr_orn, tgt_orn, t)

        return Pose(next_pos, tuple(next_orn))

    def _policy(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> Action:
        del memory  # unused
        assert isinstance(state, utils.PyBulletState)
        robot.set_joints(state.joint_positions)
        current_pose, target_pose, finger_status = (
            get_current_and_target_pose_and_finger_status(state, objects, params)
        )
        ee_action = _step_toward_pose(current_pose, target_pose)
        try:
            joint_positions = robot.inverse_kinematics(
                ee_action, validate=False, set_joints=True
            )
        except InverseKinematicsError:
            raise utils.OptionExecutionFailure("Inverse kinematics failed.")
        # Handle fingers.
        if finger_status == "open":
            finger_delta = finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -finger_action_nudge_magnitude
        state = cast(utils.PyBulletState, state)
        finger_position = state.joint_positions[robot.left_finger_joint_idx]
        f_action = finger_position + finger_delta
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        action_arr = np.clip(
            action_arr, robot.action_space.low, robot.action_space.high
        )
        assert robot.action_space.contains(action_arr)
        return Action(action_arr)

    def _terminal(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = get_current_and_target_pose_and_finger_status(
            state, objects, params
        )
        pos_dist_sq = np.sum(
            np.square(np.subtract(current_pose.position, target_pose.position))
        )
        curr_orn = np.array(current_pose.orientation)
        tgt_orn = np.array(target_pose.orientation)
        dot = float(np.clip(np.dot(curr_orn, tgt_orn), -1.0, 1.0))
        orn_dist_sq = (2.0 * np.arccos(abs(dot))) ** 2
        pose_done = (pos_dist_sq + orn_dist_sq) < move_to_pose_tol
        if not pose_done:
            return False
        if extra_terminal is not None:
            return extra_terminal(state, objects, params)
        return True

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=_terminal,
    )


def create_rotate_end_effector_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]
    ],
    rotate_tol: float,
    max_angular_vel: float,
    finger_action_nudge_magnitude: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for rotating the
    end effector about the z-axis only, by directly incrementing the last arm
    joint (joint index left_finger_joint_idx - 1). No IK is used. The target
    orientation must differ from the current one only in yaw."""

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Rotate end effector option " + f"not implemented for robot {robot_name}."
    )

    # Index of the last arm joint (the wrist rotation joint on the panda).
    last_arm_joint_idx = robot.left_finger_joint_idx - 1

    def _delta_yaw(current_orn: Tuple, target_orn: Tuple) -> float:
        """Shortest-path yaw delta from current to target orientation."""
        curr_yaw = euler_from_quaternion(current_orn)[2]
        tgt_yaw = euler_from_quaternion(target_orn)[2]
        return float((tgt_yaw - curr_yaw + np.pi) % (2 * np.pi) - np.pi)

    def _policy(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> Action:
        del memory  # unused
        assert isinstance(state, utils.PyBulletState)
        current_pose, target_pose, finger_status = (
            get_current_and_target_pose_and_finger_status(state, objects, params)
        )
        # Clamp the yaw step to max_angular_vel.
        delta = _delta_yaw(current_pose.orientation, target_pose.orientation)
        delta = np.clip(delta, -max_angular_vel, max_angular_vel)
        # Start from the current joint positions and only change the last
        # arm joint and the fingers.
        joint_positions = list(state.joint_positions)
        joint_positions[last_arm_joint_idx] += delta
        # Handle the fingers. Fingers drift if left alone.
        if finger_status == "open":
            finger_delta = finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -finger_action_nudge_magnitude
        finger_position = state.joint_positions[robot.left_finger_joint_idx]
        f_action = finger_position + finger_delta
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        action_arr = np.clip(
            action_arr, robot.action_space.low, robot.action_space.high
        )
        assert robot.action_space.contains(action_arr)
        return Action(action_arr)

    def _terminal(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = get_current_and_target_pose_and_finger_status(
            state, objects, params
        )
        delta = _delta_yaw(current_pose.orientation, target_pose.orientation)
        return delta**2 < rotate_tol

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=_terminal,
    )


def create_change_fingers_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_val: Callable[
        [State, Sequence[Object], Array], Tuple[float, float]
    ],
    max_vel_norm: float,
    grasp_tol: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for changing the
    robot fingers, given a function that takes in the current state, objects,
    and parameters, and returns the current and target finger joint values."""

    assert robot.get_name() in _SUPPORTED_ROBOTS, (
        "Change fingers option not " + f"implemented for robot {robot.get_name()}."
    )

    def _policy(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> Action:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(state, objects, params)
        f_delta = target_val - current_val
        f_delta = np.clip(f_delta, -max_vel_norm, max_vel_norm)
        f_action = current_val + f_delta
        # Don't change the rest of the joints.
        state = cast(utils.PyBulletState, state)
        target = np.array(state.joint_positions, dtype=np.float32)
        target[robot.left_finger_joint_idx] = f_action
        target[robot.right_finger_joint_idx] = f_action
        # This clipping is needed sometimes for the joint limits.
        target = np.clip(target, robot.action_space.low, robot.action_space.high)
        assert robot.action_space.contains(target)
        return Action(target)

    def _terminal(
        state: State, memory: Dict, objects: Sequence[Object], params: Array
    ) -> bool:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(state, objects, params)
        squared_dist = (target_val - current_val) ** 2
        return squared_dist < grasp_tol

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=_terminal,
    )
