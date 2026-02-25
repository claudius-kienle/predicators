"""Ground-truth options for the pybullet_coffee environment."""

from functools import partial
from typing import ClassVar, Dict, List, Sequence, Set, Tuple
from pybullet_utils.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
)

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import (
    create_change_fingers_option,
    create_move_end_effector_to_pose_option,
    create_move_end_effector_to_pose_and_orientation_option,
    create_rotate_end_effector_option,
)
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import (
    Action,
    Array,
    Object,
    ParameterizedOption,
    Predicate,
    State,
    Type,
)


class PyBulletCoffeeGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet_coffee environment."""

    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_coffee"}

    @classmethod
    def get_options(
        cls,
        env_name: str,
        types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        action_space: Box,
    ) -> Set[ParameterizedOption]:

        _, pybullet_robot, _ = PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        jug_type = types["jug"]
        machine_type = types["machine"]
        cup_type = types["cup"]

        def get_current_fingers(state: State) -> float:
            (robot,) = state.get_objects(robot_type)
            return PyBulletCoffeeEnv.fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers")
            )

        def open_fingers_func(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers
            return current, target

        def close_fingers_func(
            state: State, objects: Sequence[Object], params: Array, percent: float
        ) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            f_open = pybullet_robot.open_fingers
            f_closed = pybullet_robot.closed_fingers
            target = f_open + percent * (f_closed - f_open)

            return current, target

        # MoveToTwistJug - Move to above the jug to prepare for twisting
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0,))
        RotateItemUntilHandleAccessible = utils.LinearChainParameterizedOption(
            "RotateItemUntilHandleAccessible",
            [
                # move above jug
                cls._create_coffee_move_to_jug_top_option(
                    name="MoveEndEffectorToJugTop",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # move down around jug
                cls._create_move_relative_to_jug_option(
                    name="MoveEndEffectorDownToTwist",
                    z_offset=-0.03,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                create_change_fingers_option(
                    pybullet_robot,
                    "CloseFingers",
                    option_types,
                    params_space,
                    partial(close_fingers_func, percent=0.25),
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # rotate item
                cls._create_rotate_item_until_handle_accessible_option(
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # open fingers
                create_change_fingers_option(
                    pybullet_robot,
                    "OpenFingers",
                    option_types,
                    params_space,
                    open_fingers_func,
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # move up
                cls._create_move_relative_to_jug_option(
                    name="MoveEndEffectorBackUp",
                    z_offset=0.1,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        # PickJug - Pick up the jug by the handle
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0,))
        PickJug = utils.LinearChainParameterizedOption(
            "PickJug",
            [
                cls._create_rotate_to_abs_z_rotation(
                    abs_z_rotation=0,
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # open fingers
                # Move to above the jug handle
                cls._create_move_to_jug_handle(
                    name="MoveEndEffectorToPreGraspJug",
                    z_offset=0.1,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Open fingers
                create_change_fingers_option(
                    pybullet_robot,
                    "OpenFingers",
                    option_types,
                    params_space,
                    open_fingers_func,
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # Move to grasp the handle
                cls._create_move_to_jug_handle(
                    name="MoveEndEffectorToGraspJug",
                    z_offset=0,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Close fingers
                create_change_fingers_option(
                    pybullet_robot,
                    "CloseFingers",
                    option_types,
                    params_space,
                    partial(close_fingers_func, percent=1.0),
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # Lift the jug
                cls._create_coffee_lift_jug(
                    name="LiftJug",
                    z_offset=0.15,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        # PlaceJugInMachine - Place the jug in the coffee machine dispense area
        option_types = [robot_type, jug_type, machine_type]
        params_space = Box(0, 1, (0,))
        PlaceJugInMachine = utils.LinearChainParameterizedOption(
            "PlaceJugInMachine",
            [
                # Move above the dispense area
                cls._create_coffee_move_to_dispense_area_option(
                    name="MoveEndEffectorToPrePlace",
                    z_offset=0.08,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Move down to place
                cls._create_coffee_move_to_dispense_area_option(
                    name="MoveEndEffectorToPlace",
                    z_offset=0,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Open fingers
                create_change_fingers_option(
                    pybullet_robot,
                    "OpenFingers",
                    option_types,
                    params_space,
                    open_fingers_func,
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # Move back up
                cls._create_coffee_move_to_dispense_area_option(
                    name="MoveEndEffectorBackUp",
                    z_offset=0.1,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        CupFilled = predicates["CupFilled"]

        # TurnMachineOn - Press the button to turn the machine on
        option_types = [robot_type, machine_type]
        params_space = Box(0, 1, (0,))
        TurnMachineOnAndFill = cls._create_press_button_option(
            pybullet_robot=pybullet_robot,
            option_types=option_types,
            params_space=params_space,
        )

        # Pour - Pour from the jug into a cup
        option_types = [robot_type, jug_type, cup_type]
        params_space = Box(0, 1, (0,))
        PourSomeLiquid = utils.LinearChainParameterizedOption(
            "PourSomeLiquid",
            [
                # Move to above the cup
                cls._create_coffee_move_to_pour_position_option(
                    name="MoveEndEffectorToPrePour",
                    z_offset=PyBulletCoffeeEnv.pour_z_offset + 0.05,
                    tilt=0.0,  # No tilt while moving
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Move down to pour position
                cls._create_coffee_move_to_pour_position_option(
                    name="MoveEndEffectorToPour",
                    z_offset=PyBulletCoffeeEnv.pour_z_offset,
                    tilt=0.0,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Tilt to pour (using a special tilt option)
                cls._create_tilt_to_pour_option(
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    CupFilled=CupFilled,
                ),
                # Tilt back (untilt)
                cls._create_tilt_back_option(
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # # Move back up
                cls._create_coffee_move_to_pour_position_option(
                    name="MoveEndEffectorBackUp",
                    z_offset=0.18 + PyBulletCoffeeEnv.z_lb,
                    tilt=0.0,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        return {
            RotateItemUntilHandleAccessible,
            PickJug,
            PlaceJugInMachine,
            TurnMachineOnAndFill,
            PourSomeLiquid,
        }

    @classmethod
    def _create_coffee_move_to_jug_top_option(
        cls,
        name: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the top of the jug for twisting."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            current_pose = cls.get_current_pose(state, robot)

            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_z = (
                PyBulletCoffeeEnv.z_lb + PyBulletCoffeeEnv.jug_height + cls._offset_z
            )
            target_pose = Pose((jug_x, jug_y, jug_z), current_pose.orientation)

            return current_pose, target_pose, "open"

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def get_current_pose(cls, state: State, robot: Object) -> Pose:
        """Helper function to get the current pose of the robot's end effector."""
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        robot_z = state.get(robot, "z")
        ez = state.get(robot, "wrist")
        ey = state.get(robot, "tilt")

        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()
        ex, _, _ = euler_from_quaternion(home_orn)
        current_pose = Pose(
            (robot_x, robot_y, robot_z), tuple(quaternion_from_euler(ex, ey, ez))
        )
        return current_pose

    @classmethod
    def _create_move_to_jug_handle(
        cls,
        name: str,
        z_offset: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the jug handle position with a specified z offset."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            current_pose = cls.get_current_pose(state, robot)

            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_z = (
                PyBulletCoffeeEnv.z_lb + PyBulletCoffeeEnv.jug_height - 0.04 + z_offset
            )

            jug_x -= 0.04  # handle offset
            jug_y += 0.015

            target_pose = Pose((jug_x, jug_y, jug_z), current_pose.orientation)

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_coffee_lift_jug(
        cls,
        name: str,
        z_offset: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the jug handle position with a specified z offset."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            current_pose = cls.get_current_pose(state, robot)

            robot_x = current_pose.position[0]
            robot_y = current_pose.position[1]

            jug_z = PyBulletCoffeeEnv.z_lb + PyBulletCoffeeEnv.jug_height + z_offset

            target_pose = Pose((robot_x, robot_y, jug_z), current_pose.orientation)

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_move_relative_to_jug_option(
        cls,
        name: str,
        z_offset: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the jug handle position with a specified z offset."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            assert jug.name == "juggy"
            current_pose = cls.get_current_pose(state, robot)

            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            target_pose = Pose(
                (
                    jug_x,
                    jug_y,
                    PyBulletCoffeeEnv.z_lb + PyBulletCoffeeEnv.jug_height + z_offset,
                ),
                current_pose.orientation,
            )

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_coffee_move_to_dispense_area_option(
        cls,
        name: str,
        z_offset: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the dispense area of the coffee machine."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, machine = objects
            del machine
            current_pose = cls.get_current_pose(state, robot)

            m_x = PyBulletCoffeeEnv.machine_x
            m_y = PyBulletCoffeeEnv.machine_y

            d_area_x = m_x - 0.15
            d_area_y = m_y
            d_area_z = PyBulletCoffeeEnv.z_lb + 0.10 + z_offset

            target_pose = Pose(
                (d_area_x, d_area_y, d_area_z),
                current_pose.orientation,
            )

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_coffee_move_to_pour_position_option(
        cls,
        name: str,
        z_offset: float,
        tilt: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the pour position above a cup."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, cup = objects
            del jug  # Position derived from cup
            current_pose = cls.get_current_pose(state, robot)

            cup_x = state.get(cup, "x")
            cup_y = state.get(cup, "y")

            pour_x = cup_x - PyBulletCoffeeEnv.pour_x_offset
            pour_y = cup_y + PyBulletCoffeeEnv.pour_y_offset
            pour_z = z_offset

            target_pose = Pose((pour_x, pour_y, pour_z), current_pose.orientation)

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_press_button_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to press the machine button."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, machine = objects
            del machine  # Button position is fixed
            current_pose = cls.get_current_pose(state, robot)

            target_pose = Pose(
                (
                    PyBulletCoffeeEnv.button_x,
                    PyBulletCoffeeEnv.button_y,
                    PyBulletCoffeeEnv.button_z,
                ),
                current_pose.orientation,
            )

            return current_pose, target_pose, "open"

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            "TurnMachineOnAndFill",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_rotate_to_abs_z_rotation(
        cls,
        abs_z_rotation: float,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to rotate the jug until its handle is accessible.

        This option rotates the jug in place so that the handle faces the robot,
        making it easier to grasp.
        """

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, _ = objects
            current_pose = cls.get_current_pose(state, robot)
            ex, ey, ez = euler_from_quaternion(current_pose.orientation)

            goal_rot_jug = abs_z_rotation
            delta_rot = (goal_rot_jug - ez) % (2 * np.pi)
            # print(delta_rot)
            # if abs(delta_rot - np.pi) < 0.1:
            #     delta_rot = 0
            # if delta_rot > np.pi:
            #     delta_rot -= 2 * np.pi

            goal_euler = (ex, ey, ez + delta_rot)
            goal_orn = quaternion_from_euler(*goal_euler)

            target_pose = Pose(current_pose.position, goal_orn.tolist())

            return current_pose, target_pose, "closed"

        return create_move_end_effector_to_pose_and_orientation_option(
            pybullet_robot,
            "RotateItem",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            0.3,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_rotate_item_until_handle_accessible_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to rotate the jug until its handle is accessible.

        This option rotates the jug in place so that the handle faces the robot,
        making it easier to grasp.
        """

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            current_pose = cls.get_current_pose(state, robot)

            jug_rot = state.get(jug, "rot")

            goal_rot_jug = PyBulletCoffeeEnv.jug_pickable_rot

            delta_rot = -(goal_rot_jug - jug_rot + np.pi) % (2 * np.pi) - np.pi

            if abs(delta_rot) < 0.1:
                delta_rot = 0

            curr_euler = euler_from_quaternion(current_pose.orientation)
            goal_euler = (curr_euler[0], curr_euler[1], curr_euler[2] + delta_rot)
            goal_orn = quaternion_from_euler(*goal_euler)

            target_pose = Pose(
                current_pose.position,
                goal_orn.tolist(),
            )

            return current_pose, target_pose, "closed"

        return create_rotate_end_effector_option(
            pybullet_robot,
            "RotateItem",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            0.3,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_tilt_to_pour_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
        CupFilled: Predicate,
    ) -> ParameterizedOption:
        """Create an option to tilt the jug to pour by rotating around the y-axis.

        Analogous to the twist action (which rotates around z), but targets the
        y-axis to achieve a pouring tilt.
        """

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, cup = objects
            del jug, cup  # unused
            current_pose = cls.get_current_pose(state, robot)
            ex, ey, ez = euler_from_quaternion(current_pose.orientation)

            goal_ey = PyBulletCoffeeEnv.pour_tilt_angle
            delta_rot = (goal_ey - ey) % (2 * np.pi)
            if abs(delta_rot - np.pi) < 0.1:
                delta_rot = 0

            goal_euler = (ex, ey + delta_rot, ez)
            goal_orn = quaternion_from_euler(*goal_euler)

            target_pose = Pose(current_pose.position, goal_orn.tolist())
            return current_pose, target_pose, "closed"

        def _cup_filled_terminal(
            state: State, objects: Sequence[Object], params: Array
        ) -> bool:
            del params  # unused
            _, _, cup = objects
            return CupFilled.holds(state, [cup])

        return create_move_end_effector_to_pose_and_orientation_option(
            pybullet_robot,
            "TiltToPour",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            0.3,
            cls._finger_action_nudge_magnitude,
            extra_terminal=_cup_filled_terminal,
        )

    @classmethod
    def _create_tilt_back_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to tilt the jug back to upright."""

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, cup = objects
            del jug, cup  # unused
            current_pose = cls.get_current_pose(state, robot)
            ex, ey, ez = euler_from_quaternion(current_pose.orientation)

            goal_ey = 0.0
            delta_rot = (goal_ey - ey) % (2 * np.pi)
            if abs(delta_rot - np.pi) < 0.1:
                delta_rot = 0

            goal_euler = (ex, ey + delta_rot, ez)
            goal_orn = quaternion_from_euler(*goal_euler)

            target_pose = Pose(current_pose.position, goal_orn.tolist())
            return current_pose, target_pose, "closed"

        return create_move_end_effector_to_pose_and_orientation_option(
            pybullet_robot,
            "TiltBack",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            0.3,
            cls._finger_action_nudge_magnitude,
        )
