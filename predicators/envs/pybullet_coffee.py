"""A PyBullet version of Coffee."""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
from pybullet_utils.transformations import euler_from_quaternion, quaternion_from_euler

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import (
    SingleArmPyBulletRobot,
    create_single_arm_pybullet_robot,
)
from predicators.settings import CFG
from predicators.structs import (
    Action,
    Array,
    EnvironmentTask,
    Object,
    Predicate,
    State,
    Type,
)


class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain."""

    # Parameters that aren't important enough to need to clog up settings.py
    _finger_action_tol: ClassVar[float] = 5e-4

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.2)
    _table_orientation: ClassVar[Quaternion] = (0.0, 0.0, 0.0, 1.0)

    # Override coordinate bounds from CoffeeEnv to use PyBullet-appropriate scales
    # (in meters, similar to PyBulletBlocksEnv).
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    y_lb: ClassVar[float] = 0.3
    y_ub: ClassVar[float] = 1.0
    z_lb: ClassVar[float] = (
        0.36  # Table surface height (table box height 0.4m, centered at z=0)
    )
    z_ub: ClassVar[float] = 1.2  # Reduced from 1.5 to be reachable

    # Recompute derived values based on new bounds.
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2.0
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2.0
    robot_init_z: ClassVar[float] = 0.7  # Use a reachable height similar to blocks env

    # Scale machine dimensions to fit new coordinate system.
    machine_x_len: ClassVar[float] = 0.15
    machine_y_len: ClassVar[float] = 0.15
    machine_z_len: ClassVar[float] = 0.3
    machine_x: ClassVar[float] = x_ub + 0.1
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - 0.55

    # Scale jug dimensions.
    jug_radius: ClassVar[float] = 0.04
    jug_height: ClassVar[float] = 0.13
    jug_init_x_lb: ClassVar[float] = x_lb
    jug_init_x_ub: ClassVar[float] = x_ub
    jug_init_y_lb: ClassVar[float] = machine_y + 0.5
    jug_init_y_ub: ClassVar[float] = machine_y + 0.65
    jug_handle_height: ClassVar[float] = 3 * jug_height / 4

    # Scale cup dimensions.
    cup_radius: ClassVar[float] = 0.03
    cup_init_x_lb: ClassVar[float] = x_lb
    cup_init_x_ub: ClassVar[float] = x_ub
    cup_init_y_lb: ClassVar[float] = machine_y + 0.15
    cup_init_y_ub: ClassVar[float] = machine_y + 0.5
    cup_capacity_lb: ClassVar[float] = 0.06
    cup_capacity_ub: ClassVar[float] = 0.13
    cup_liquid_radius_scale: ClassVar[float] = 0.8
    cup_liquid_min_height: ClassVar[float] = 0.002

    jug_liquid_radius_scale: ClassVar[float] = 0.6
    jug_liquid_height_scale: ClassVar[float] = 0.35
    liquid_color: ClassVar[Tuple[float, float, float, float]] = (0.36, 0.2, 0.08, 0.9)

    # Recompute button position.
    button_x: ClassVar[float] = machine_x - 0.07
    button_y: ClassVar[float] = machine_y
    button_z: ClassVar[float] = z_lb + 0.15
    button_radius: ClassVar[float] = 0.02

    # Dispense area.
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 1.5 * jug_radius

    # Update tolerances for new scale.
    grasp_position_tol: ClassVar[float] = 0.0001
    dispense_tol: ClassVar[float] = 0.1
    pour_pos_tol: ClassVar[float] = 0.05 ** 2 # squared distance
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.15
    jug_pickable_rot: ClassVar[float] = np.pi

    # Update simulation parameters.
    pour_x_offset: ClassVar[float] = -1.5 * (cup_radius + jug_radius)
    pour_y_offset: ClassVar[float] = -0.015
    pour_z_offset: ClassVar[float] = (
        1.4 * (cup_capacity_ub + jug_height - jug_handle_height) + z_lb
    )
    tilt_ub: ClassVar[float] = np.pi / 6
    pour_tilt_angle: ClassVar[float] = tilt_ub
    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0
    max_position_vel: ClassVar[float] = 0.25
    max_angular_vel: ClassVar[float] = np.pi / 4  # Same as tilt_ub from parent
    max_finger_vel: ClassVar[float] = 0.1

    # Camera parameters.
    _camera_yaw: ClassVar[float] = -20.0
    _camera_pitch: ClassVar[float] = -40
    _camera_distance: ClassVar[float] = 0.8
    _camera_target: ClassVar[Pose3D] = ((x_lb + x_ub) / 2.0, (y_lb + y_ub) / 2.0, 0.32)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self._JugPickable = Predicate(
            "JugPickable", [self._jug_type], self._JugPickable_holds
        )

        # We track the correspondence between PyBullet object IDs and Object
        # instances. This correspondence changes with the task.
        self._cup_id_to_cup: Dict[int, Object] = {}

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._CupFilled,
            self._Holding,
            self._OnTable,
            self._HandEmpty,
            self._JugPickable,
            self._JugFilled,
            self._JugInMachine,
            self._MachineOn,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        match CFG.pybullet_detailed_goal_predicates:
            case "all":
                return {
                    self._CupFilled,
                    self._Holding,
                    self._OnTable,
                    self._HandEmpty,
                    self._RobotAboveCup,
                    self._JugAboveCup,
                    self._NotAboveCup,
                    self._PressingButton,
                    self._NotSameCup,
                    self._JugPickable,
                }
            case "std":
                return {self._CupFilled}

    def _JugPickable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The jug is pickable when on-table and handle rotation is accessible."""
        (jug,) = objects
        if not self._OnTable_holds(state, [jug]) and not self._JugInMachine_holds(state, [jug, self._machine]):
            return False
        jug_rot = state.get(jug, "rot")
        rot_error = (jug_rot - self.jug_pickable_rot + np.pi) % (2 * np.pi) - np.pi
        return abs(rot_error) < self.pick_jug_rot_tol

    def _robot_jug_above_cup(self, state: State, cup: Object) -> bool:
        if not self._Holding_holds(state, [self._robot, self._jug]):
            return False
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = state.get(self._robot, "z") - self.jug_handle_height
        jug_pos = (jug_x, jug_y, jug_z)
        cup_x = state.get(cup, "x")
        cup_y = state.get(cup, "y")
        pour_pos = (
            cup_x - self.pour_x_offset,
            cup_y + self.pour_y_offset,
            self.pour_z_offset,
        )
        sq_dist_to_pour = np.linalg.norm(np.subtract(jug_pos, pour_pos))
        return bool(sq_dist_to_pour < self.pour_pos_tol)

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
            rgbaColor=[0.9, 0.9, 0.9, 1.0],
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
        bodies["jug_liquid_id"] = cls._create_liquid_body(
            radius=cls.jug_radius * cls.jug_liquid_radius_scale,
            height=cls.jug_height * cls.jug_liquid_height_scale,
            physics_client_id=physics_client_id,
        )

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
        cup_liquid_ids = []
        cup_liquid_radius = cls.cup_radius * cls.cup_liquid_radius_scale
        for _ in range(num_cups):
            cup_liquid_ids.append(
                cls._create_liquid_body(
                    radius=cup_liquid_radius,
                    height=cls.cup_liquid_min_height,
                    physics_client_id=physics_client_id,
                )
            )
        bodies["cup_liquid_ids"] = cup_liquid_ids

        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _create_coffee_machine(cls, physics_client_id: int) -> int:
        """Create the coffee machine from SAPIEN mesh."""
        # Load the SAPIEN coffee machine URDF
        # The URDF has rotation applied and bounding box: min=[-0.52, -0.68, -0.59], max=[0.52, 0.69, 0.84]
        # With globalScaling=0.1, the height is ~0.14m
        scale = 0.006

        # Position the machine so its bottom sits on the table
        machine_z = cls.z_lb + 0.04

        machine_id = p.loadURDF(
            utils.get_env_asset_path("urdf/coffee_machine/custom/machine.urdf"),
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

    @classmethod
    def _create_liquid_body(
        cls, radius: float, height: float, physics_client_id: int
    ) -> int:
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=cls.liquid_color,
            physicsClientId=physics_client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=[0.0, 0.0, -10.0],
            baseOrientation=cls._default_orn,
            physicsClientId=physics_client_id,
        )

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._machine_id = pybullet_bodies["machine_id"]
        self._jug_id = pybullet_bodies["jug_id"]
        self._jug_liquid_id = pybullet_bodies["jug_liquid_id"]
        self._cup_ids = pybullet_bodies["cup_ids"]
        self._cup_liquid_ids = pybullet_bodies["cup_liquid_ids"]

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
        q = self.get_robot_ee_home_orn()
        ex, _, _ = euler_from_quaternion(q)
        ey = state.get(self._robot, "tilt")  # Use tilt from state
        ez = state.get(self._robot, "wrist")  # Use wrist rotation from state
        qx, qy, qz, qw = quaternion_from_euler(ex, ey, ez)

        # Map finger state: open_fingers corresponds to 1, closed to 0.
        f = state.get(self._robot, "fingers")
        # Normalize to PyBullet finger joint values.
        # if fingers > (self.open_fingers + self.closed_fingers) / 2:
        #     f = self._pybullet_robot.open_fingers
        # else:
        #     f = self._pybullet_robot.closed_fingers

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

        # Reset coffee-specific latent state variables from the symbolic state.
        self._jug_filled_state = float(state.get(self._jug, "is_filled"))
        self._machine_on_state = float(state.get(self._machine, "is_on"))

        # Reset jug based on the state.
        jug_x = state.get(self._jug, "x")
        jug_y = state.get(self._jug, "y")
        jug_z = state.get(self._jug, "z")
        jug_rot = state.get(self._jug, "rot")

        # Convert 2D rotation to quaternion (rotation around z-axis).
        jug_orn = p.getQuaternionFromEuler([0, 0, jug_rot])

        p.resetBasePositionAndOrientation(
            self._jug_id,
            [jug_x, jug_y, jug_z],
            jug_orn,
            physicsClientId=self._physics_client_id,
        )

        jug_held = state.get(self._jug, "is_held") > 0.5

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
            cup_z = state.get(cup_obj, "z")

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

            self._update_liquid_visuals(state)

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
        # rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        s = self._pybullet_robot.get_state()

        rx, ry, rz = s[:3]
        ex, ey, ez = euler_from_quaternion(s[3:7])
        rf = s[-1]

        # Map PyBullet finger joint to state finger value.
        # open_f = self._pybullet_robot.open_fingers
        # closed_f = self._pybullet_robot.closed_fingers
        fingers = rf
        # if rf > (open_f + closed_f) * 1 / 2:
        #     fingers = self.open_fingers
        # else:
        #     fingers = self.closed_fingers

        # In the coffee env, robot also has tilt and wrist.
        # For PyBullet version, we keep these constant or derive from pose.
        # tilt = self.tilt_lb  # Default to no tilt
        tilt = ey
        # wrist = self.robot_init_wrist  # Default wrist rotation
        wrist = ez

        state_dict[self._robot] = np.array(
            [rx, ry, rz, tilt, wrist, fingers], dtype=np.float32
        )

        # Get jug state.
        (jug_x, jug_y, jug_z), jug_orn_quat = p.getBasePositionAndOrientation(
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
            [jug_x, jug_y, jug_z, jug_rot, float(jug_held), jug_filled], dtype=np.float32
        )

        # Get machine state.
        if hasattr(self, "_machine_on_state"):
            machine_on = self._machine_on_state
        else:
            machine_on = 0.0
        state_dict[self._machine] = np.array(
            [self.machine_x, self.machine_y, self.z_lb, machine_on], dtype=np.float32
        )

        # Get cup states.
        for cup_id, cup in self._cup_id_to_cup.items():
            (cup_x, cup_y, cup_z), _ = p.getBasePositionAndOrientation(
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
                [cup_x, cup_y, cup_z, capacity, target, current], dtype=np.float32
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
            assert self._JugPickable.holds(next_state, [self._jug]), "Jug should be pickable when filled."

        # Check for pouring: jug must be held and tilted to tilt_ub.
        jug_held = self._Holding_holds(next_state, [self._robot, self._jug])
        jug_filled = (
            self._jug_filled_state > 0.5
            if hasattr(self, "_jug_filled_state")
            else False
        )
        if jug_held and jug_filled:
            tilt = next_state.get(self._robot, "tilt")
            if abs(abs(tilt) - self.tilt_ub) < self.pour_angle_tol:
                cup = self._get_cup_to_pour(next_state)
                if cup is not None:
                    current_liquid = next_state.get(cup, "current_liquid")
                    new_liquid = current_liquid + self.pour_velocity
                    capacity = next_state.get(cup, "capacity_liquid")
                    if new_liquid <= capacity:
                        # Update the live observation so _get_state() picks up
                        # this change via _current_state.
                        self._current_observation.set(cup, "current_liquid", new_liquid)

        # Update the current observation with modified state variables.
        # We need to rebuild the state with updated values.
        self._current_observation = self._get_state()
        self._update_liquid_visuals(self._current_observation)

        return self._current_observation.copy()

    def render(
        self, action: Optional[Action] = None, caption: Optional[str] = None
    ) -> List[np.ndarray]:  # pragma: no cover
        del action, caption  # unused

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id,
        )

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id,
        )

        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._physics_client_id,
        )

        rgb_array = np.array(px).reshape((height, width, 4))[:, :, :3]

        obj_to_body_id = self._get_render_label_body_map()
        if obj_to_body_id:
            pil_img = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
            draw = ImageDraw.Draw(pil_img)
            view = np.array(view_matrix, dtype=np.float64).reshape((4, 4), order="F")
            proj = np.array(proj_matrix, dtype=np.float64).reshape((4, 4), order="F")
            view_proj = proj @ view

            for obj, body_id in obj_to_body_id.items():
                if obj == self._robot:
                    robot_base_pos = p.getBasePositionAndOrientation(
                        self._pybullet_robot.robot_id,
                        physicsClientId=self._physics_client_id,
                    )[0]
                    label_pos = (
                        float(robot_base_pos[0]),
                        float(robot_base_pos[1]),
                        float(robot_base_pos[2]),
                    )
                else:
                    label_pos = self._get_body_center(body_id)
                pixel_xy = self._project_point_to_pixel(
                    point_world=label_pos,
                    width=width,
                    height=height,
                    view_proj_matrix=view_proj,
                )
                if pixel_xy is None:
                    continue
                px_x, px_y = pixel_xy
                text = str(obj)
                text_bbox = draw.textbbox((px_x, px_y), text)
                pad = 2
                bg_box = (
                    text_bbox[0] - pad,
                    text_bbox[1] - pad,
                    text_bbox[2] + pad,
                    text_bbox[3] + pad,
                )
                draw.rectangle(bg_box, fill=(0, 0, 0))
                draw.text((px_x, px_y), text, fill=(255, 255, 255))

            table_text = "table:table"
            table_bbox = draw.textbbox((0, 0), table_text)
            table_w = table_bbox[2] - table_bbox[0]
            table_h = table_bbox[3] - table_bbox[1]
            table_x = int((width - table_w) / 2) + 20
            table_y = max(0, height - table_h - 20)
            pad = 2
            table_bg = (
                table_x - pad,
                table_y - pad,
                table_x + table_w + pad,
                table_y + table_h + pad,
            )
            draw.rectangle(table_bg, fill=(0, 0, 0))
            draw.text((table_x, table_y), table_text, fill=(255, 255, 255))

            rgb_array = np.array(pil_img)

        return [rgb_array]

    def _get_render_label_body_map(self) -> Dict[Object, int]:
        """Map each Coffee object to its corresponding PyBullet body ID."""
        obj_to_body_id: Dict[Object, int] = {
            self._robot: self._pybullet_robot.robot_id,
            self._machine: self._machine_id,
            self._jug: self._jug_id,
        }
        for cup_id, cup_obj in self._cup_id_to_cup.items():
            obj_to_body_id[cup_obj] = cup_id
        return obj_to_body_id

    def _get_body_center(self, body_id: int) -> Tuple[float, float, float]:
        """Get the geometric center of a body from its axis-aligned bounds."""
        aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self._physics_client_id)
        center = (np.array(aabb_min) + np.array(aabb_max)) / 2.0
        return float(center[0]), float(center[1]), float(center[2])

    def _project_point_to_pixel(
        self,
        point_world: Tuple[float, float, float],
        width: int,
        height: int,
        view_proj_matrix: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """Project a world-space point into image pixel coordinates."""
        point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        clip = view_proj_matrix @ point_h
        w = clip[3]
        if abs(w) < 1e-8:
            return None

        ndc = clip[:3] / w
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None

        px_x = int((ndc[0] * 0.5 + 0.5) * width)
        px_y = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
        if px_x < 0 or px_x >= width or px_y < 0 or px_y >= height:
            return None
        return px_x, px_y

    def _update_liquid_visuals(self, state: State) -> None:
        """Update cup and jug liquid visuals from symbolic state values."""
        oov_x, oov_y = self._out_of_view_xy

        jug_filled = state.get(self._jug, "is_filled") > 0.5
        if jug_filled:
            (jug_x, jug_y, jug_z), jug_orn = p.getBasePositionAndOrientation(
                self._jug_id, physicsClientId=self._physics_client_id
            )
            liquid_z = jug_z + 0.04 - 0.1 * self.jug_height
            p.resetBasePositionAndOrientation(
                self._jug_liquid_id,
                [jug_x + 0.01, jug_y + 0.02, liquid_z],
                jug_orn,
                physicsClientId=self._physics_client_id,
            )
        else:
            p.resetBasePositionAndOrientation(
                self._jug_liquid_id,
                [oov_x, oov_y, -10.0],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )

        for cup_idx, cup_id in enumerate(self._cup_ids):
            liquid_id = self._cup_liquid_ids[cup_idx]
            cup_obj = self._cup_id_to_cup.get(cup_id)
            if cup_obj is None:
                p.resetBasePositionAndOrientation(
                    liquid_id,
                    [oov_x, oov_y, -10.0],
                    self._default_orn,
                    physicsClientId=self._physics_client_id,
                )
                continue

            capacity = state.get(cup_obj, "capacity_liquid")
            current = state.get(cup_obj, "current_liquid")
            if capacity <= 0 or current <= 0:
                p.resetBasePositionAndOrientation(
                    liquid_id,
                    [oov_x, oov_y, -10.0],
                    self._default_orn,
                    physicsClientId=self._physics_client_id,
                )
                continue

            fill_ratio = np.clip(current / capacity, 0.0, 1.0)
            liquid_height = max(fill_ratio * capacity, self.cup_liquid_min_height)
            (cup_x, cup_y, _), cup_orn = p.getBasePositionAndOrientation(
                cup_id, physicsClientId=self._physics_client_id
            )
            liquid_z = self.z_lb + 0.07 + liquid_height / 2.0

            p.changeVisualShape(
                liquid_id,
                linkIndex=-1,
                rgbaColor=self.liquid_color,
                physicsClientId=self._physics_client_id,
            )
            p.resetBasePositionAndOrientation(
                liquid_id,
                [cup_x, cup_y, liquid_z],
                cup_orn,
                physicsClientId=self._physics_client_id,
            )

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
            # normal = np.array([1., 0., 0.], dtype=np.float32)
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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
        # return open_f if fingers_state > threshold else closed_f
        return fingers_state

    def _get_jug_z(self, state: State, jug: Object) -> float:
        """Get the z position of the jug."""
        if state.get(jug, "is_held") > 0.5:
            # If held, jug z is relative to robot position
            (robot,) = state.get_objects(self._robot_type)
            return state.get(robot, "z") - self.jug_handle_height
        else:
            # If not held, jug sits on the table
            return self.z_lb
