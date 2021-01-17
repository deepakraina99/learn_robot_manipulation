#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provide the reinforcement learning environment to learn the ofrc control in table wiping task.

The goal is to track the force along z axis (vertical to the table) by employing admittance control, meanwhile tracking
a predefined trajectory on the xy plane.
"""

"""
Action: [Position gain (Kp), Damping parameter (kd)]
Observation: [Position Error (x_e), Force Error (f_e), end-effector velocity (dx)] 
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import numpy as np
import os, time

class TargetReachingEnv(gym.Env):
    metadata = {'render.modes':['human']}
    # def __init__(self, render=True, robot_type = 'ur5', action_robot_len=6, obs_robot_len=15, reward_type=1):
    def __init__(self, render=True, robot_type = 'rrbot', action_robot_len=2, obs_robot_len=16, reward_type=1):

        """
        Initialize the environment.

        Args:
            action_robot_len (int, None): length of action space
            obs_robot_len (int, None): length of observation space
        """

        # Start the bullet physics server
        self.gui = render
        if self.gui:
            self.pid = p.connect(p.GUI)
        else:
            self.pid = p.connect(p.DIRECT)
        
        self.robot_type = robot_type
        self.max_speed = np.pi #rad/sec
        # self.dt = 1./240 #time step
        self.time_step = 0.02
        self.max_time = 100
        self.iteration = 0
        self.reward_ver = reward_type
        
        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len
        self.action_space = spaces.Box(low=np.array([-self.max_speed]*self.action_robot_len), high=np.array([self.max_speed]*self.action_robot_len), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.0]*self.obs_robot_len), high=np.array([1.0]*self.obs_robot_len), dtype=np.float32)

        self.target_position = np.array([0.0, 0.0, 0.0]) #[x,y,z]
        self.obs_state = None
        self.target_radius = 0.2

        #attributes
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.seed() #initialize a seed
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        # action = np.array([0, 0])
        # action *= 0.5 
        
        # update state
        # q = self.get_joint_positions(self.robot, self.controllable_joints)
        # q = q + action * self.dt

        # q = np.clip(q, a_min=self.joint_lower_limits, a_max=self.joint_upper_limits)

        # p.setJointMotorControlArray(self.robot, self.controllable_joints, 
        #                             controlMode=p.POSITION_CONTROL, targetPositions=q)
                    
        p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=action)
        self.iteration +=1
        if self.last_sim_time is None:
            self.last_sim_time = time.time()

        p.stepSimulation()
        # Slow down time so that the simulation matches real time
        self.slow_time()
        
        #get an observation of state
        obs_state = self.get_observation()

        # get reward        
        distance_to_target = self.euclidian_distance(self.target_position, np.asarray(p.getLinkState(self.robot, self.tool)[0]))
        # print('distance-to-target %.2f:' % (distance_to_target))

        #calculate the reward
        reward_dist = -distance_to_target
        reward_ctrl = -np.square(action).sum()
        # Reward: 0
        if self.reward_ver == 0:
            reward = reward_dist
        # Reward: 1
        if self.reward_ver == 1:
            reward = reward_dist + reward_ctrl
        # Reward: 2 or 3
        if self.reward_ver == 2 or self.reward_ver == 3:
            if not reached:
                reward = - 1.0 + reward_ctrl
            else:
                reward = 1000
        
        info = {'action': action[0], 'Position Error': distance_to_target}
       
        # reached = False
        # terminated = False
        # if distance_to_target < self.target_radius:
        #     print('Reached Goal !!')
        #     reached = True
        # if (self.iteration*self.dt) > self.max_time:
        #     print('Time Limit Reached !!')
        #     terminated = True

        # done = reached or terminated
        done = False

        return obs_state, reward, done, info
        
    def reset(self):
        print('Reset Env!!')
        self.create_world()
        self.robot, self.controllable_joints, self.tool, self.joint_lower_limits, self.joint_upper_limits  = self.load_robot()
        self.target, self.targte_position, self.target_orientation = self.load_target()
        self.iteration = 0
        self.last_sim_time = None

        return self.get_observation()
        
    def create_world(self):

        p.resetSimulation(physicsClientId=self.pid)
        # Configure camera position
        # p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.pid)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.pid)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.pid)

        # Load all models off screen and then move them into place
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.pid)

        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.pid)

        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.pid)

        # Load the base for robot
        if self.robot_type == 'ur5':
            self.load_box(position=[0,0,0.25], dimensions=(0.1,0.1,0.5), mass=0, color=(0.5,0.5,0.5,1))

        # Load the surface
        # self.load_box(position=np.array([0.6, 0., 0.25]), dimensions=(0.7, 1, 0.5), mass=1)

    def load_robot(self):
        if self.robot_type == 'ur5':
            # Load the UR5 robot
            robot = p.loadURDF(os.path.join(self.directory, 'ur', 'ur5-afc.urdf'), useFixedBase=True, basePosition=[0, 0, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.pid)
            control_joint_indices = [0, 1, 2, 3, 4, 5]
            tool_index = 6
            # Reset the joint states (change to random initial position)
            initial_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, -1.57]
        
        if self.robot_type == 'rrbot':
            # Load the UR5 robot
            robot = p.loadURDF(os.path.join(self.directory, 'rrbot', 'rrbot.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.pid)
            control_joint_indices = [1, 2]
            tool_index = 3
            # Reset the joint states (change to random initial position)
            initial_joint_positions = [0.0, 0.5]

        # Let the environment settle
        for _ in range(100):
            p.stepSimulation()

        self.reset_joint_states(robot, joint_ids=control_joint_indices,  positions=initial_joint_positions)
        joint_lower_limits, joint_upper_limits = self.get_joint_limits(robot, control_joint_indices)

        return robot, control_joint_indices, tool_index, joint_lower_limits, joint_upper_limits

    def get_joint_limits(self, body_id, joint_ids):
        lower_limits = []
        upper_limits = []
        
        for j in joint_ids:
            joint_info = p.getJointInfo(body_id, j)
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
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dimensions / 2.)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=dimensions / 2., rgbaColor=color)

        box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape,
                                   basePosition=position, baseOrientation=orientation, physicsClientId=self.pid)
        return box

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
            p.resetJointState(body_id, jointIndex=joint_id, targetValue=position)

    def get_joint_positions(self, body_id, joint_ids):
        """
        Get the position of the given joint(s).

        Args:
            body_id (int): unique body id.
            joint_ids (int, list[int]): joint id, or list of joint ids.

        Returns:
            if 1 joint:
                float: joint position [rad]
            if multiple joints:
                np.array[float[N]]: joint positions [rad]
        """
        if isinstance(joint_ids, int):
            return p.getJointState(body_id, joint_ids)[0]
        return np.asarray([state[0] for state in p.getJointStates(body_id, joint_ids)])

    def load_target(self):

        if self.robot_type == 'ur5':
            sphere_radius = 0.05
            # self.target_position = np.array([0.5, 0.1, 0.75])
            target_orientation = np.array([1, 0, 0, 0])
        elif self.robot_type == 'rrbot':
            sphere_radius = 0.1
            target_high, target_low = np.array([2.0, 0.2, 4.0]), np.array([0.1, 0.2, 0.1]) 
            # self.target_position = np.array([1.0, 0.2, 3.0])
            target_orientation = np.array([1, 0, 0, 0])

        while True:
            target_position = self.np_random.uniform(low=target_low, high=target_high)
            if np.linalg.norm(target_position-self.tool_position(self.robot, self.tool)) < 2.0:
                break
        
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=(1, 0, 0, 0.5))
        target = p.createMultiBody(baseVisualShapeIndex=visual_shape, baseMass=0., basePosition=list(target_position), physicsClientId=self.pid)
        
        return target, target_position, target_orientation
    
    def tool_position(self, body_id, tool_id):
        return np.asarray(p.getLinkState(body_id, tool_id)[0])

    def get_observation(self):
        # get current end-effector position and velocity in the task/operational space
        x = np.asarray(p.getLinkState(self.robot, self.tool)[0])
        dx = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[6])
        o = np.asarray(p.getLinkState(self.robot, self.tool)[1])
        do = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[7])

        # get end-effector velocity in task space
        tool_pose = np.concatenate((x, o))
        tool_velocity = np.concatenate((dx, do))

        return np.concatenate((self.target_position, tool_pose, tool_velocity)).ravel()

    def get_jacobian(self, q):
        r"""
        Return the full geometric Jacobian matrix :math:`J(q) = [J_{lin}(q)^T, J_{ang}(q)^T]^T`, such that:

        .. math:: v = [\dot{p}^T, \omega^T]^T = J(q) \dot{q}

        where :math:`\dot{p}` is the Cartesian linear velocity of the link, and :math:`\omega` is its angular velocity.

        Args:
            body_id (int): unique body id.
            link_id (int): link id.
            local_position (np.array[float[3]]): the point on the specified link to compute the Jacobian (in link
              local coordinates around its center of mass). If None, it will use the CoM position (in the link frame).
            q (np.array[float[N]]): joint positions of size N, where N is the number of DoFs.
            dq (np.array[float[N]]): joint velocities of size N, where N is the number of DoFs.
            des_ddq (np.array[float[N]]): desired joint accelerations of size N.

        Returns:
            np.array[float[6,N]], np.array[float[6,6+N]]: full geometric (linear and angular) Jacobian matrix. The
                number of columns depends if the base is fixed or floating.
        """
        # Note that q, dq, ddq have to be lists in PyBullet (it doesn't work with numpy arrays)
        if isinstance(q, np.ndarray):
            q = q.ravel().tolist()

        local_position = p.getLinkState(self.robot, self.tool)[2]
        dq = [0]*len(q)
        # calculate full jacobian
        lin_jac, ang_jac = p.calculateJacobian(self.robot, self.tool, localPosition=local_position,
                                                      objPositions=list(q), objVelocities=dq, objAccelerations=dq)
        return np.vstack((lin_jac, ang_jac))

    def enable_force_torque_sensor(self, body_id, joint_ids):
        
        if isinstance(joint_ids, int):
            p.enableJointForceTorqueSensor(body_id, joint_ids, 1)
        else:
            for joint_id in joint_ids:
                p.enableJointForceTorqueSensor(body_id, joint_id, 1)

    def force_torque_sensing(self, body_id, joint_ids):
        """
        Return the joint reaction forces at the given joint. Note that the torque sensor must be enabled, otherwise
        it will always return [0,0,0,0,0,0].

        Args:
            body_id (int): unique body id.
            joint_ids (int, int[N]): joint id, or list of joint ids

        Returns:
            if 1 joint:
                np.array[float[6]]: joint reaction force (fx,fy,fz,mx,my,mz) [N,Nm]
            if multiple joints:
                np.array[float[N,6]]: joint reaction forces [N, Nm]
        """
        if isinstance(joint_ids, int):
            return np.asarray(p.getJointState(body_id, joint_ids)[2])
        return np.asarray([state[2] for state in p.getJointStates(body_id, joint_ids)])

    def quaternion_error(self, quat_des, quat_cur):
        r"""
        Compute the orientation (vector) error between the current and desired quaternion; that is, it is the difference
        between :math:`q_curr` and :math:`q_des`, which is given by: :math:`\Delta q = q_{curr}^{-1} q_{des}`.
        Only the vector part is returned which can be used in PD control.

        Args:
            quat_des (np.array[float[4]]): desired quaternion [x,y,z,w]
            quat_cur (np.array[float[4]]): current quaternion [x,y,z,w]

        Returns:
            np.array[float[3]]: vector error between the current and desired quaternion
        """
        diff = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] - self.skew_matrix(quat_des[:3]).dot(quat_cur[:3])
        return diff

    def skew_matrix(self, vector):
        r"""
        Return the skew-symmetric matrix of the given vector, which allows to represents the cross product between the
        given vector and another vector, as the multiplication of the returned skew-symmetric matrix with the other
        vector.

        The skew-symmetric matrix from a 3D vector :math:`v=[x,y,z]` is given by:

        .. math::

            S(v) = \left[ \begin{array}{ccc} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \\ \end{array} \right]

        It can be shown [2] that: :math:`\dot{R}(t) = \omega(t) \times R(t) = S(\omega(t)) R(t)`, where :math:`R(t)` is
        a rotation matrix that varies as time :math:`t` goes, :math:`\omega(t)` is the angular velocity vector of frame
        :math:`R(t) with respect to the reference frame at time :math:`t`, and :math:`S(.)` is the skew operation that
        returns the skew-symmetric matrix from the given vector.

        Args:
            vector (np.array[float[3]]): 3D vector

        Returns:
            np.array[float[3,3]]: skew-symmetric matrix

        References:
            - [1] Wikipedia: https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product
            - [2] "Robotics: Modelling, Planning and Control" (sec 3.1.1), by Siciliano et al., 2010
        """
        x, y, z = np.array(vector).flatten()
        return np.array([[0., -z, y],
                        [z, 0., -x],
                        [-y, x, 0.]])

    def get_damped_least_squares_inverse(self, jacobian, damping_factor=0.01):
        r"""
        Return the damped least-squares (DLS) inverse, given by:

        .. math:: \hat{J} = J^T (JJ^T + k^2 I)^{-1}

        which can then be used to get joint velocities :math:`\dot{q}` from the cartesian velocities :math:`v`, using
        :math:`\dot{q} = \hat{J} v`.

        Args:
            jacobian (np.array[float[D,N]]): Jacobian matrix
            damping_factor (float): damping factor

        Returns:
            np.array[float[N,D]]: DLS inverse matrix
        """
        J, k = jacobian, damping_factor
        return J.T.dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

    def map_action_values(self, output_lower_limit, output_upper_limit, action_value):
        input_upper_limit, input_lower_limit = self.action_space.high[0], self.action_space.low[0]
        output = output_lower_limit + ((output_upper_limit - output_lower_limit) / (input_upper_limit - input_lower_limit)) * (action_value - input_lower_limit)
        return output

    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    # def render(self, mode='human', close=False):
    #     if not self.gui:
    #         self.gui = True
    #         p.disconnect(self.pid)
    #         self.pid = p.connect(p.GUI)

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            p.disconnect(self.pid)
            self.pid = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))

            # self.world_creation = WorldCreation(self.id, robot_type=self.robot_type, task=self.task, time_step=self.time_step, np_random=self.np_random, config=self.config)
            # self.util = Util(self.id, self.np_random)
            # # print('Physics server ID:', self.id)
