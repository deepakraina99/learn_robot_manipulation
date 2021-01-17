import os
import numpy as np
import pybullet as p
from .utils import Utils


class WorldCreation:
    def __init__(self, pid, robot_type='rrbot', time_step=0.02, np_random=None):
        self.id = pid
        self.robot_type = robot_type
        self.time_step = time_step
        self.np_random = np_random
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.utils = Utils(self.id, self.np_random)

    def create_new_world(self):
        p.resetSimulation(physicsClientId=self.id)

        # Configure camera position
        # p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

        # Load all models off screen and then move them into place
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)

        # Load the base for robot
        if self.robot_type == 'ur5':
            self.load_box(position=[0,0,0.25], dimensions=(0.1,0.1,0.5), mass=0, color=(0.5,0.5,0.5,1))

        robot, controllable_joints, tool_index, joint_lower_limits, joint_upper_limits  = self.load_robot()
        target, target_pos, target_orient = self.load_target(robot, tool_index)

        return robot, controllable_joints, tool_index, joint_lower_limits, joint_upper_limits, target, target_pos, target_orient

    def load_box(self, position, orientation=(0, 0, 0, 1), mass=1., dimensions=(1., 1., 1.), color=None):
        """
        Load a box in the world (only available in the simulator).

        Args:
            position (float[3]): position of the box in the Cartesian world space (in meters)
            orientation (float[4]): orientation of the box using quaternion [x,y,z,w].
            mass (float): mass of the box (in kg). If mass = 0, the box won't move even if there is a collision.
            dimensions (float[3]): dimensions of the box (in meter)
            color (int[4], None): color of the box for red, green, blue, and alpha, each in range [0,1]

        Returns:
            int, Body: unique id of the box in the world, or the box body
        """
        dimensions = np.asarray(dimensions)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dimensions / 2., physicsClientId=self.id)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=dimensions / 2., rgbaColor=color, physicsClientId=self.id)

        box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
                                basePosition=position, baseOrientation=orientation, physicsClientId=self.id)
        return box

    def load_robot(self):

        if self.robot_type == 'ur5':
            # Load the UR5 robot
            robot = p.loadURDF(os.path.join(self.directory, 'ur', 'ur5-afc.urdf'), useFixedBase=True, basePosition=[0, 0, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
            control_joint_indices = [0, 1, 2, 3, 4, 5]
            tool_index = 6
            # Reset the joint states (change to random initial position)
            initial_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, -1.57]
        
        if self.robot_type == 'rrbot':
            # Load the UR5 robot
            robot = p.loadURDF(os.path.join(self.directory, 'rrbot', 'rrbot.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
            control_joint_indices = [1, 2]
            tool_index = 3
            # Reset the joint states (change to random initial position)
            initial_joint_positions = [0.0, 0.5]

        # Let the environment settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        self.reset_joint_states(robot, joint_ids=control_joint_indices,  positions=initial_joint_positions)
        joint_lower_limits, joint_upper_limits = self.get_joint_limits(robot, control_joint_indices)

        return robot, control_joint_indices, tool_index, joint_lower_limits, joint_upper_limits

    def get_joint_limits(self, body_id, joint_ids):
        lower_limits = []
        upper_limits = []
        
        for j in joint_ids:
            joint_info = p.getJointInfo(body_id, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)

        lower_limits = np.array(lower_limits)
        upper_limits = np.array(upper_limits)
        
        return lower_limits, upper_limits

    def reset_joint_states(self, body_id, joint_ids, positions):
        """
        Reset the joint states. It is best only to do this at the start, while not running the simulation:
        `reset_joint_state` overrides all physics simulation.

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint indices where each joint index is between [0..num_joints(body_id)]
            positions (float, list[float], np.array[float]): the joint position(s) (angle in radians [rad] or
              position [m])
        """
        # reset the joint states
        for i, joint_id in enumerate(joint_ids):
            position = positions[i]
            p.resetJointState(body_id, jointIndex=joint_id, targetValue=position, physicsClientId=self.id)

    def load_target(self, robot_id, tool_id):

        if self.robot_type == 'ur5':
            sphere_radius = 0.05
            # self.target_position = np.array([0.5, 0.1, 0.75])
            target_high, target_low = np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, 0.5]) 
            target_orientation = np.array([1, 0, 0, 0])
        elif self.robot_type == 'rrbot':
            sphere_radius = 0.1
            target_high, target_low = np.array([2.0, 0.2, 4.0]), np.array([-2.0, 0.2, 0.0]) 
            target_orientation = np.array([1, 0, 0, 0])

        while True:
            target_position = self.np_random.uniform(low=target_low, high=target_high)
            if np.linalg.norm(target_position-self.robot_base_position()) < 1.0:
                break
        
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=(1, 0, 0, 0.5), physicsClientId=self.id)
        target = p.createMultiBody(baseVisualShapeIndex=visual_shape, baseMass=0., basePosition=list(target_position), physicsClientId=self.id)
        
        return target, target_position, target_orientation

    def robot_base_position(self):
        if self.robot_type == 'rrbot':
            return np.array([0, 0, 1])
        elif self.robot_type == 'ur5':
            return np.array([0, 0, 0.5])
        