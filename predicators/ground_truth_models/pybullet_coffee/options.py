"""Ground-truth options for the pybullet_coffee environment."""

from typing import ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import (
    create_change_fingers_option,
    create_move_end_effector_to_pose_option,
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
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers
            return current, target

        # MoveToTwistJug - Move to above the jug to prepare for twisting
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0,))
        MoveToTwistJug = utils.LinearChainParameterizedOption(
            "MoveToTwistJug",
            [
                cls._create_coffee_move_to_jug_top_option(
                    name="MoveEndEffectorToJugTop",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        # RotateItemUntilHandleAccessible - Rotate the jug until handle is accessible
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0,))  # No parameters needed
        RotateItemUntilHandleAccessible = cls._create_rotate_item_until_handle_accessible_option(
            pybullet_robot=pybullet_robot,
            option_types=option_types,
            params_space=params_space,
        )

        # PickJug - Pick up the jug by the handle
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0,))
        PickJug = utils.LinearChainParameterizedOption(
            "PickJug",
            [
                # Move to above the jug handle
                cls._create_coffee_move_to_jug_handle_option(
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
                cls._create_coffee_move_to_jug_handle_option(
                    name="MoveEndEffectorToGraspJug",
                    z_offset=cls._offset_z,
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
                    close_fingers_func,
                    CFG.pybullet_max_vel_norm,
                    PyBulletCoffeeEnv.grasp_position_tol,
                ),
                # Lift the jug
                cls._create_coffee_move_to_jug_handle_option(
                    name="LiftJug",
                    z_offset=0.2,
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
                # Move to above the dispense area
                cls._create_coffee_move_to_dispense_area_option(
                    name="MoveEndEffectorToPrePlace",
                    z=PyBulletCoffeeEnv.robot_init_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Move down to place
                cls._create_coffee_move_to_dispense_area_option(
                    name="MoveEndEffectorToPlace",
                    z=PyBulletCoffeeEnv.z_lb + cls._offset_z,
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
                    z=PyBulletCoffeeEnv.robot_init_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        # TurnMachineOn - Press the button to turn the machine on
        option_types = [robot_type, machine_type]
        params_space = Box(0, 1, (0,))
        TurnMachineOn = cls._create_press_button_option(
            pybullet_robot=pybullet_robot,
            option_types=option_types,
            params_space=params_space,
        )

        # Pour - Pour from the jug into a cup
        option_types = [robot_type, jug_type, cup_type]
        params_space = Box(0, 1, (0,))
        Pour = utils.LinearChainParameterizedOption(
            "Pour",
            [
                # Move to above the cup
                cls._create_coffee_move_to_pour_position_option(
                    name="MoveEndEffectorToPrePour",
                    z_offset=0.15,
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
                ),
                # Tilt back (untilt)
                cls._create_tilt_back_option(
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
                # Move back up
                cls._create_coffee_move_to_pour_position_option(
                    name="MoveEndEffectorBackUp",
                    z_offset=0.15,
                    tilt=0.0,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                ),
            ],
        )

        return {
            MoveToTwistJug,
            RotateItemUntilHandleAccessible,
            PickJug,
            PlaceJugInMachine,
            TurnMachineOn,
            Pour,
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
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            current_pose = Pose((robot_x, robot_y, robot_z), home_orn)

            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_z = PyBulletCoffeeEnv.jug_height
            target_pose = Pose((jug_x, jug_y, jug_z), home_orn)

            return current_pose, target_pose, "open"

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            CFG.pybullet_max_vel_norm,
            cls._move_to_pose_tol,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_coffee_move_to_jug_handle_option(
        cls,
        name: str,
        z_offset: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the jug handle position with a specified z offset."""
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug = objects
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            current_pose = Pose((robot_x, robot_y, robot_z), home_orn)

            # Get jug handle position from CoffeeEnv logic
            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_rot = state.get(jug, "rot")
            jug_radius = PyBulletCoffeeEnv.jug_radius
            jug_handle_height = PyBulletCoffeeEnv.jug_handle_height

            # Handle is at radius distance in the direction of rotation
            handle_x = jug_x + jug_radius * np.cos(jug_rot + np.pi / 2)
            handle_y = jug_y + jug_radius * np.sin(jug_rot + np.pi / 2)
            handle_z = PyBulletCoffeeEnv.z_lb + jug_handle_height + z_offset

            target_pose = Pose((handle_x, handle_y, handle_z), home_orn)

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            CFG.pybullet_max_vel_norm,
            cls._move_to_pose_tol,
            cls._finger_action_nudge_magnitude,
        )

    @classmethod
    def _create_coffee_move_to_dispense_area_option(
        cls,
        name: str,
        z: float,
        finger_status: str,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Move to the dispense area of the coffee machine."""
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, machine = objects
            del machine  # Position is fixed in env constants
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            current_pose = Pose((robot_x, robot_y, robot_z), home_orn)

            # If jug is held, adjust z position accounting for handle
            if state.get(jug, "is_held") > 0.5:
                target_z = z + PyBulletCoffeeEnv.jug_handle_height
            else:
                target_z = z

            target_pose = Pose(
                (
                    PyBulletCoffeeEnv.dispense_area_x,
                    PyBulletCoffeeEnv.dispense_area_y,
                    target_z,
                ),
                home_orn,
            )

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            CFG.pybullet_max_vel_norm,
            cls._move_to_pose_tol,
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
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, jug, cup = objects
            del jug  # Position derived from cup
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            current_pose = Pose((robot_x, robot_y, robot_z), home_orn)

            cup_x = state.get(cup, "x")
            cup_y = state.get(cup, "y")

            pour_x = cup_x + PyBulletCoffeeEnv.pour_x_offset
            pour_y = cup_y + PyBulletCoffeeEnv.pour_y_offset
            pour_z = z_offset + PyBulletCoffeeEnv.z_lb

            target_pose = Pose((pour_x, pour_y, pour_z), home_orn)

            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            CFG.pybullet_max_vel_norm,
            cls._move_to_pose_tol,
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
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
            state: State, objects: Sequence[Object], params: Array
        ) -> Tuple[Pose, Pose, str]:
            del params  # unused
            robot, machine = objects
            del machine  # Button position is fixed
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            robot_z = state.get(robot, "z")
            current_pose = Pose((robot_x, robot_y, robot_z), home_orn)

            target_pose = Pose(
                (
                    PyBulletCoffeeEnv.button_x,
                    PyBulletCoffeeEnv.button_y,
                    PyBulletCoffeeEnv.button_z,
                ),
                home_orn,
            )

            return current_pose, target_pose, "open"

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            "TurnMachineOn",
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            CFG.pybullet_max_vel_norm,
            cls._move_to_pose_tol,
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
        home_orn = PyBulletCoffeeEnv.get_robot_ee_home_orn()
        rotation_speed = 0.5  # Radians per step

        def _policy(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> Action:
            # Rotate the jug by applying force/torque or by direct manipulation
            # For now, maintain position as rotation would need physics manipulation
            robot, jug = objects
            current_joints = state.simulator_state
            return Action(np.array(current_joints, dtype=np.float32))

        def _terminal(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> bool:
            # Check if handle is accessible (facing the robot)
            robot, jug = objects
            robot_x = state.get(robot, "x")
            robot_y = state.get(robot, "y")
            jug_x = state.get(jug, "x")
            jug_y = state.get(jug, "y")
            jug_rot = state.get(jug, "rot")
            
            # Vector from jug to robot
            dx = robot_x - jug_x
            dy = robot_y - jug_y
            angle_to_robot = np.arctan2(dy, dx)
            
            # Handle is at jug_rot + pi/2
            handle_angle = jug_rot + np.pi / 2
            
            # Normalize angles to [-pi, pi]
            angle_diff = np.abs(((angle_to_robot - handle_angle + np.pi) % (2 * np.pi)) - np.pi)
            
            # Terminal if handle is roughly facing the robot (within 30 degrees)
            return angle_diff < np.pi / 6

        return ParameterizedOption(
            "RotateItemUntilHandleAccessible",
            types=option_types,
            params_space=params_space,
            policy=_policy,
            initiable=lambda s, m, o, p: True,
            terminal=_terminal,
        )

    @classmethod
    def _create_tilt_to_pour_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to tilt the jug to pour.

        Note: This is a simplified version. Full implementation would require
        proper orientation control in PyBullet.
        """

        def _policy(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> Action:
            # Simplified: maintain position while "pouring"
            # Full implementation would change end-effector orientation
            robot, jug, cup = objects
            current_joints = state.simulator_state
            return Action(np.array(current_joints, dtype=np.float32))

        def _terminal(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> bool:
            # Check if cup is filled (simulated by the environment)
            robot, jug, cup = objects
            cup_current = state.get(cup, "current_liquid")
            cup_target = state.get(cup, "target_liquid")
            return cup_current >= cup_target - 1e-3

        return ParameterizedOption(
            "TiltToPour",
            types=option_types,
            params_space=params_space,
            policy=_policy,
            initiable=lambda s, m, o, p: True,
            terminal=_terminal,
        )

    @classmethod
    def _create_tilt_back_option(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        option_types: List[Type],
        params_space: Box,
    ) -> ParameterizedOption:
        """Create an option to tilt the jug back to upright."""

        def _policy(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> Action:
            # Simplified: maintain position
            robot, jug, cup = objects
            current_joints = state.simulator_state
            return Action(np.array(current_joints, dtype=np.float32))

        def _terminal(
            state: State, memory: Dict, objects: Sequence[Object], params: Array
        ) -> bool:
            # Immediately terminal for simplified version
            return True

        return ParameterizedOption(
            "TiltBack",
            types=option_types,
            params_space=params_space,
            policy=_policy,
            initiable=lambda s, m, o, p: True,
            terminal=_terminal,
        )
