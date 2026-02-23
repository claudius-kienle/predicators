"""A PyBullet version of Coffee."""

import logging
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import (
    SingleArmPyBulletRobot,
    create_single_arm_pybullet_robot,
)
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, State


class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain."""

    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.65, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0.0, 0.0, 0.0, 1.0)

    # Override coordinate bounds from CoffeeEnv to use PyBullet-appropriate scales
    # (in meters, similar to PyBulletBlocksEnv).
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    z_lb: ClassVar[float] = (
        0.16  # Table surface height (table box height 0.4m, centered at z=0)
    )
    z_ub: ClassVar[float] = 1.0  # Reduced from 1.5 to be reachable

    # Recompute derived values based on new bounds.
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2.0
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2.0
    robot_init_z: ClassVar[float] = 0.7  # Use a reachable height similar to blocks env

    # Scale machine dimensions to fit new coordinate system.
    machine_x_len: ClassVar[float] = 0.15
    machine_y_len: ClassVar[float] = 0.15
    machine_z_len: ClassVar[float] = 0.3
    machine_x: ClassVar[float] = x_ub - machine_x_len / 2 + 0.45
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - 0.55

    # Scale jug dimensions.
    jug_radius: ClassVar[float] = 0.04
    jug_height: ClassVar[float] = 0.12
    jug_init_x_lb: ClassVar[float] = machine_x - 0.2
    jug_init_x_ub: ClassVar[float] = machine_x - 0.1
    jug_init_y_lb: ClassVar[float] = machine_y + 0.2
    jug_init_y_ub: ClassVar[float] = machine_y + 0.6
    jug_handle_height: ClassVar[float] = 3 * jug_height / 4

    # Scale cup dimensions.
    cup_radius: ClassVar[float] = 0.03
    cup_init_x_lb: ClassVar[float] = machine_x - 0.2
    cup_init_x_ub: ClassVar[float] = machine_x - 0.1
    cup_init_y_lb: ClassVar[float] = machine_y + 0.2
    cup_init_y_ub: ClassVar[float] = machine_y + 0.6
    cup_capacity_lb: ClassVar[float] = 0.06
    cup_capacity_ub: ClassVar[float] = 0.12

    # Recompute button position.
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[float] = machine_y - machine_y_len / 2
    button_z: ClassVar[float] = z_lb + 3 * machine_z_len / 4

    # Dispense area.
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 1.5 * jug_radius

    # Update tolerances for new scale.
    grasp_position_tol: ClassVar[float] = 0.05
    dispense_tol: ClassVar[float] = 0.1
    pour_pos_tol: ClassVar[float] = 0.1
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.15

    # Update simulation parameters.
    pour_x_offset: ClassVar[float] = 1.5 * (cup_radius + jug_radius)
    pour_y_offset: ClassVar[float] = cup_radius
    pour_z_offset: ClassVar[float] = 1.1 * (
        cup_capacity_ub + jug_height - jug_handle_height
    )
    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0
    max_position_vel: ClassVar[float] = 0.25
    max_angular_vel: ClassVar[float] = np.pi / 4  # Same as tilt_ub from parent
    max_finger_vel: ClassVar[float] = 0.1

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # We track the correspondence between PyBullet object IDs and Object
        # instances. This correspondence changes with the task.
        self._cup_id_to_cup: Dict[int, Object] = {}

    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle coffee-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load table.
        table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            table_id,
            cls._table_pose,
            cls._table_orientation,
            physicsClientId=physics_client_id,
        )
        # Scale the table to fit the coffee environment dimensions.
        p.changeVisualShape(
            table_id,
            -1,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert using_gui, "using_gui must be True to use pybullet_draw_debug."
            # Draw the workspace boundaries for clarity.
            table_z = 0.625  # Table height
            p.addUserDebugLine(
                [cls.x_lb, cls.y_lb, table_z],
                [cls.x_ub, cls.y_lb, table_z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_lb, cls.y_ub, table_z],
                [cls.x_ub, cls.y_ub, table_z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_lb, cls.y_lb, table_z],
                [cls.x_lb, cls.y_ub, table_z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )
            p.addUserDebugLine(
                [cls.x_ub, cls.y_lb, table_z],
                [cls.x_ub, cls.y_ub, table_z],
                [1.0, 0.0, 0.0],
                lineWidth=5.0,
                physicsClientId=physics_client_id,
            )

        # Create coffee machine.
        machine_id = cls._create_coffee_machine(physics_client_id)
        bodies["machine_id"] = machine_id

        # Create jug.
        jug_id = cls._create_coffee_jug(physics_client_id)
        bodies["jug_id"] = jug_id

        # Create cups. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove cups from the workspace
        # based on which ones are in the state.
        num_cups = max(max(CFG.coffee_num_cups_train), max(CFG.coffee_num_cups_test))
        cup_ids = []
        for i in range(num_cups):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            cup_id = cls._create_coffee_cup(color, physics_client_id)
            cup_ids.append(cup_id)
        bodies["cup_ids"] = cup_ids

        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _create_coffee_machine(cls, physics_client_id: int) -> int:
        """Create the coffee machine from SAPIEN mesh."""
        # Load the SAPIEN coffee machine URDF
        # The URDF has rotation applied and bounding box: min=[-0.52, -0.68, -0.59], max=[0.52, 0.69, 0.84]
        # With globalScaling=0.1, the height is ~0.14m
        scale = 0.20
        machine_height = 0.18 * scale / 0.1  # Approximate height after scaling

        # Position the machine so its bottom sits on the table
        machine_z = cls.z_lb + machine_height / 2

        machine_id = p.loadURDF(
            utils.get_env_asset_path("urdf/coffee_machine/103046/mobility.urdf"),
            basePosition=[cls.machine_x, cls.machine_y, machine_z],
            baseOrientation=cls._default_orn,
            useFixedBase=True,
            globalScaling=scale,
            physicsClientId=physics_client_id,
        )

        return machine_id

    @classmethod
    def _create_coffee_jug(cls, physics_client_id: int) -> int:
        """Create the coffee jug using custom OBJ model."""
        # Get path to jug mesh
        jug_mesh_path = utils.get_env_asset_path("urdf/jug/jug.obj")

        scale = [0.01 * 0.3 for _ in range(3)]

        # Create collision shape from OBJ
        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=jug_mesh_path,
            meshScale=scale,  # Adjust scale as needed
            physicsClientId=physics_client_id,
        )

        # Create visual shape from OBJ
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=jug_mesh_path,
            meshScale=scale,
            physicsClientId=physics_client_id,
        )

        # Create multibody
        jug_id = p.createMultiBody(
            baseMass=cls._obj_mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[cls.jug_init_x_lb, cls.jug_init_y_lb, cls.z_lb],
            baseOrientation=cls._default_orn,
            physicsClientId=physics_client_id,
        )

        p.changeDynamics(
            jug_id,
            linkIndex=-1,
            lateralFriction=cls._obj_friction,
            physicsClientId=physics_client_id,
        )

        return jug_id

    @classmethod
    def _create_coffee_cup(
        cls, color: Tuple[float, float, float, float], physics_client_id: int
    ) -> int:
        """Create a coffee cup using YCB mug model."""
        # Load YCB mug for cups
        cup_id = p.loadURDF(
            "urdf/mug.urdf",
            basePosition=[cls.cup_init_x_lb, cls.cup_init_y_lb, cls.z_lb],
            baseOrientation=cls._default_orn,
            globalScaling=0.8,
            physicsClientId=physics_client_id,
        )

        # Try to change color if possible
        p.changeVisualShape(
            cup_id, -1, rgbaColor=color, physicsClientId=physics_client_id
        )

        p.changeDynamics(
            cup_id,
            linkIndex=-1,
            mass=cls._obj_mass * 0.5,
            lateralFriction=cls._obj_friction,
            physicsClientId=physics_client_id,
        )

        return cup_id

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._machine_id = pybullet_bodies["machine_id"]
        self._jug_id = pybullet_bodies["jug_id"]
        self._cup_ids = pybullet_bodies["cup_ids"]

    @classmethod
    def _create_pybullet_robot(cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose(
            (cls.robot_init_x, cls.robot_init_y, cls.robot_init_z), robot_ee_orn
        )
        return create_single_arm_pybullet_robot(
            CFG.pybullet_robot, physics_client_id, ee_home
        )

    def _extract_robot_state(self, state: State) -> Array:
        """Extract robot state from State for PyBullet."""
        # The orientation is fixed in this environment (no tilt in pybullet).
        qx, qy, qz, qw = self.get_robot_ee_home_orn()

        # Map finger state: open_fingers corresponds to 1, closed to 0.
        fingers = state.get(self._robot, "fingers")
        # Normalize to PyBullet finger joint values.
        if fingers > (self.open_fingers + self.closed_fingers) / 2:
            f = self._pybullet_robot.open_fingers
        else:
            f = self._pybullet_robot.closed_fingers

        return np.array(
            [
                state.get(self._robot, "x"),
                state.get(self._robot, "y"),
                state.get(self._robot, "z"),
                qx,
                qy,
                qz,
                qw,
                f,
            ],
            dtype=np.float32,
        )

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_coffee"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle coffee-specific resetting."""
        super()._reset_state(state)

        # Reset jug based on the state.
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = self._get_jug_z(state, self._jug)
        jug_rot = state.get(self._jug, "rot")

        # Convert 2D rotation to quaternion (rotation around z-axis).
        jug_orn = p.getQuaternionFromEuler([0, 0, jug_rot])

        p.resetBasePositionAndOrientation(
            self._jug_id,
            [jug_x, jug_y, jug_z + self.jug_height / 2.0],
            jug_orn,
            physicsClientId=self._physics_client_id,
        )

        # Update jug color based on whether it's filled.
        jug_filled = state.get(self._jug, "is_filled") > 0.5
        jug_held = state.get(self._jug, "is_held") > 0.5
        if jug_filled and jug_held:
            color = [0.0, 0.0, 0.6, 1.0]  # Dark blue
        elif jug_filled:
            color = [0.5, 0.7, 1.0, 1.0]  # Light blue
        elif jug_held:
            color = [0.0, 0.5, 0.0, 1.0]  # Dark green
        else:
            color = [0.7, 0.9, 0.7, 1.0]  # Light green

        p.changeVisualShape(
            self._jug_id,
            linkIndex=-1,
            rgbaColor=color,
            physicsClientId=self._physics_client_id,
        )

        # Reset cups based on the state.
        cup_objs = state.get_objects(self._cup_type)
        self._cup_id_to_cup = {}
        # All cups use the same average height for the cylinder geometry
        cup_height = (self.cup_capacity_lb + self.cup_capacity_ub) / 2.0

        for i, cup_obj in enumerate(cup_objs):
            cup_id = self._cup_ids[i]
            self._cup_id_to_cup[cup_id] = cup_obj

            cup_x = state.get(cup_obj, "x")
            cup_y = state.get(cup_obj, "y")
            # Position cup center so bottom sits on table
            cup_z = self.z_lb + cup_height / 2.0

            p.resetBasePositionAndOrientation(
                cup_id,
                [cup_x, cup_y, cup_z],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

            # Update cup visual to show capacity (by scaling height).
            # Note: This is a simplification; ideally we'd show liquid level.

        # Check if we're holding the jug.
        if jug_held:
            self._force_grasp_jug()

        # For any cups not involved, put them out of view.
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(cup_objs), len(self._cup_ids)):
            cup_id = self._cup_ids[i]
            assert cup_id not in self._cup_id_to_cup
            p.resetBasePositionAndOrientation(
                cup_id,
                [oov_x, oov_y, i * 0.1],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        state_dict = {}

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()

        # Map PyBullet finger joint to state finger value.
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        if rf > (open_f + closed_f) / 2:
            fingers = self.open_fingers
        else:
            fingers = self.closed_fingers

        # In the coffee env, robot also has tilt and wrist.
        # For PyBullet version, we keep these constant or derive from pose.
        tilt = self.tilt_lb  # Default to no tilt
        wrist = self.robot_init_wrist  # Default wrist rotation

        state_dict[self._robot] = np.array(
            [rx, ry, rz, tilt, wrist, fingers], dtype=np.float32
        )

        # Get jug state.
        (jug_x, jug_y, jug_z_center), jug_orn_quat = p.getBasePositionAndOrientation(
            self._jug_id, physicsClientId=self._physics_client_id
        )

        # Convert quaternion to euler to get z-rotation.
        jug_euler = p.getEulerFromQuaternion(jug_orn_quat)
        jug_rot = jug_euler[2]  # z-axis rotation

        # Determine if jug is held.
        jug_held = self._held_obj_id == self._jug_id

        # Get current jug filled state (we need to preserve this from simulation).
        # This is tricky - we need to look at the previous state or track separately.
        # For now, we'll check the visual color to infer the state.
        jug_filled = 0.0
        if hasattr(self, "_jug_filled_state"):
            jug_filled = self._jug_filled_state
        else:
            jug_filled = 0.0

        state_dict[self._jug] = np.array(
            [jug_x, jug_y, jug_rot, float(jug_held), jug_filled], dtype=np.float32
        )

        # Get machine state.
        if hasattr(self, "_machine_on_state"):
            machine_on = self._machine_on_state
        else:
            machine_on = 0.0
        state_dict[self._machine] = np.array([machine_on], dtype=np.float32)

        # Get cup states.
        for cup_id, cup in self._cup_id_to_cup.items():
            (cup_x, cup_y, cup_z_center), _ = p.getBasePositionAndOrientation(
                cup_id, physicsClientId=self._physics_client_id
            )

            # Retrieve capacity and liquid info from previous state.
            if hasattr(self, "_current_state") and self._current_state is not None:
                capacity = self._current_state.get(cup, "capacity_liquid")
                target = self._current_state.get(cup, "target_liquid")
                current = self._current_state.get(cup, "current_liquid")
            else:
                # Default values if no previous state.
                capacity = (self.cup_capacity_lb + self.cup_capacity_ub) / 2.0
                target = capacity * self.cup_target_frac
                current = 0.0

            state_dict[cup] = np.array(
                [cup_x, cup_y, capacity, target, current], dtype=np.float32
            )

        joint_positions = self._pybullet_robot.get_joints()
        state = utils.PyBulletState(state_dict, simulator_state=joint_positions)

        return state

    def step(self, action: Action) -> State:
        """Override step to add coffee-specific simulation logic."""
        # Get the state before the step.
        prev_state = self._current_observation.copy()

        # Call parent step to handle PyBullet physics.
        next_state = super().step(action)

        # Now handle coffee-specific logic that's not naturally handled by physics.

        # Check if button should be pressed.
        pressing_button = self._PressingButton_holds(
            next_state, [self._robot, self._machine]
        )
        machine_was_on = (
            self._machine_on_state > 0.5
            if hasattr(self, "_machine_on_state")
            else False
        )

        if pressing_button and not machine_was_on:
            # Turn on the machine.
            self._machine_on_state = 1.0

        # Check if jug should be filled.
        jug_in_machine = self._JugInMachine_holds(
            next_state, [self._jug, self._machine]
        )
        machine_on = (
            self._machine_on_state > 0.5
            if hasattr(self, "_machine_on_state")
            else False
        )

        if jug_in_machine and machine_on:
            self._jug_filled_state = 1.0
            # Update jug color.
            p.changeVisualShape(
                self._jug_id,
                linkIndex=-1,
                rgbaColor=[0.5, 0.7, 1.0, 1.0],
                physicsClientId=self._physics_client_id,
            )

        # Update the current observation with modified state variables.
        # We need to rebuild the state with updated values.
        self._current_observation = self._get_state()

        return self._current_observation.copy()

    def simulate(self, state: State, action: Action) -> State:
        """Use the CoffeeEnv simulate logic for abstract planning."""
        # For abstract simulation (used in planning), use the parent coffee logic.
        # This maintains compatibility with the non-PyBullet version.
        return CoffeeEnv.simulate(self, state, action)

    def _get_tasks(
        self, num: int, num_cups_lst: List[int], rng: np.random.Generator
    ) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, num_cups_lst, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return list of PyBullet IDs for objects that can be held."""
        # In coffee environment, only the jug can be held.
        assert self._jug_id is not None
        return [self._jug_id]

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        """Get expected finger normals for grasping."""
        if CFG.pybullet_robot == "panda":
            # Gripper rotated 90deg so parallel to x-axis.
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # Gripper parallel to y-axis.
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:  # pragma: no cover
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

    def _force_grasp_jug(self) -> None:
        """Force the robot to grasp the jug."""
        # The jug should already be held based on the state.
        held_obj_id = self._detect_held_object()
        if held_obj_id == self._jug_id:
            self._held_obj_id = self._jug_id
            self._create_grasp_constraint()
        else:
            # If not detecting jug as held, force it.
            self._held_obj_id = self._jug_id
            self._create_grasp_constraint()

    @classmethod
    def fingers_state_to_joint(
        cls, pybullet_robot: SingleArmPyBulletRobot, fingers_state: float
    ) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either open_fingers or closed_fingers.
        Transform them to be either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        threshold = (cls.open_fingers + cls.closed_fingers) / 2
        return open_f if fingers_state > threshold else closed_f

    def _get_jug_z(self, state: State, jug: Object) -> float:
        """Get the z position of the jug."""
        if state.get(jug, "is_held") > 0.5:
            # If held, jug z is relative to robot position
            (robot,) = state.get_objects(self._robot_type)
            return state.get(robot, "z") - self.jug_handle_height
        else:
            # If not held, jug sits on the table
            return self.z_lb
