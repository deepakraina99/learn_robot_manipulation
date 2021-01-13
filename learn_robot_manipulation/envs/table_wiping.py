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
import os
# from learn_force_control.envs.utils import Body
# from pyrobolearn.robots import Body
# from pyrobolearn.simulators import Bullet


class TableWipingEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, action_robot_len=2, obs_robot_len=19, render=True):

        """
        Initialize the environment.

        Args:
            action_robot_len (int, None): length of action space
            obs_robot_len (int, None): length of observation space
        """

        # Start the bullet physics server
        self.pid = p.connect(p.GUI)
        # self.pid = p.connect(p.DIRECT)

        self.gui = render
        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len
        self.action_space = spaces.Box(low=np.array([-1.0]*self.action_robot_len), high=np.array([1.0]*self.action_robot_len), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.0]*self.obs_robot_len), high=np.array([1.0]*self.obs_robot_len), dtype=np.float32)

        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.time_step = 1./240
                    
    def step(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        
        kp_act = action[0] # gain for position 
        kf_act = action[1] # stiffness parametter for admittance control

        # hyperparameters
        kp_base, kp_range = 500, 250 
        kf_base, kf_range = 500000, 10000

        kp = self.map_action_values(kp_base-kp_range, kp_base+kp_range, kp_act)
        kf = self.map_action_values(kf_base-kf_range, kf_base+kf_range, kf_act)

        kd = 2*np.sqrt(kp) # critical damping
        
        mf = 1
        damp_ratio = 1
        bf = 2*damp_ratio*np.sqrt(kf*mf)

        # get current end-effector position and velocity in the task/operational space
        x = np.asarray(p.getLinkState(self.robot, self.tool)[0])
        dx = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[6])
        o = np.asarray(p.getLinkState(self.robot, self.tool)[1])
        do = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[7])

        # get current joint states
        q = np.asarray([state[0] for state in p.getJointStates(self.robot, self.controllable_joints)])

        # get jacobian
        J = self.get_jacobian(q)
        # Pseudo-inverse: \hat{J} = J^T (JJ^T + k^2 I)^{-1}
        Jp = self.get_damped_least_squares_inverse(J)

        # get force torque sensor values
        F = self.force_torque_sensing(self.robot, joint_ids=6)
        Fz_current = F[2] 
        Fz_error = Fz_current - self.Fz_desired  # record the current error
        # print('Fz_error: ', Fz_error)

        # set the M\D\K parameters by heuristic method, these parameters may have a good result
        # M, D, K = 1, 9500, 500000
        M, D, K = mf, bf, kf
        dt = self.time_step

        # the formula is theta_x(k) = Fc(k)*Ts^2+Bd*Ts*theta_x(k-1)+Md*(2*theta_x(k-1)-theta_x(k-2))/(Md+Bd*Ts+Kd*Ts^2)
        numerator = Fz_error * np.square(dt) + D * dt * self.detx[1] + M * (2 * self.detx[1] - self.detx[2])
        denominator = M + D*dt + K*np.square(dt)
        self.detx_ = numerator / denominator
        # print (detx_)
        self.detx[2] = self.detx[1]
        self.detx[1] = self.detx[0]
        self.detx[0] = self.detx_

        zzz = self.target_position[2] + self.detx[0]

        # circle_center the the centre of the circle trajectory on the table
        # define gains
        # kp = 500  # 5 if velocity control, 50 if position control
        # kd = 5  # 2*np.sqrt(kp)
        
        # define amplitude and angular velocity when moving the sphere
        w = 0.001
        r = 0.1

        # self.target_position = np.array([0.5, 0.1, 0.75])
        # self.target_position = np.array([0.5, 0.1, 0.75-0.000005*self.iteration])
        self.target_position = np.array([self.circle_center[0] - r * np.sin(w * self.iteration + np.pi / 2), 
                                        self.circle_center[1] + r * np.cos(w * self.iteration + np.pi / 2),
                                        zzz])
        dv = kp * (self.target_position - x) - kd * dx  # compute the other direction tracking error term
        dw = kp * self.quaternion_error(self.target_orientation, o) - kd * do
        
        # evaluate damped-least-squares IK
        dq = Jp.dot(np.hstack((dv, dw)))

        q = q + dq * dt

        p.setJointMotorControlArray(self.robot, self.controllable_joints, controlMode=p.POSITION_CONTROL, targetPositions=q)
        self.iteration +=1
        
        p.stepSimulation()

        obs = self.get_observation()

        # Reward
        pose_error_max, force_error_max = 1, 1
        reward_pose_error = - np.linalg.norm((np.concatenate((self.target_position, self.target_orientation)) - np.concatenate((x, o))))
        reward_force_error = - Fz_error     
        pose_weight = 0.5
        force_weight = 0.5
        reward = pose_weight*reward_pose_error + force_weight*reward_force_error
        reward = -50 if Fz_current > self.Fz_desired else 0
    
        done = False if Fz_current < 50 else True
        
        info = {'Force (z)': Fz_current, 'Kp': kp, 'Kf': kf}

        return obs, reward, done, info

    def reset(self):
        self.create_world()
        self.robot, self.controllable_joints, self.tool = self.load_robot(robot='ur5')
        self.iteration = 0
        self.detx = np.array([0.0, 0.0, 0.0])
        # r = 0.1 needs to be taken care below
        self.circle_center = np.array([self.target_position[0]+0.1, self.target_position[1]]) # the center of the trajectory
        self.enable_force_torque_sensor(self.robot, joint_ids=6)
        self.Fz_desired = 50
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
        self.load_box(position=[0,0,0.25], dimensions=(0.1,0.1,0.5), mass=0, color=(0.5,0.5,0.5,1))

        # Load the surface
        self.load_box(position=np.array([0.6, 0., 0.25]), dimensions=(0.7, 1, 0.5), mass=1)

        # Load a visual sphere in the world (only available in the simulator)
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=(1, 0, 0, 0.5))
        self.target = p.createMultiBody(baseVisualShapeIndex=visual_shape, baseMass=0., basePosition=[0.6, 0.1, 0.5])
        self.target_position = np.array([0.5, 0.1, 0.75])
        self.target_orientation = np.array([1, 0, 0, 0])

    def load_robot(self, robot='ur5'):

        # Load the UR5 robot
        robot = p.loadURDF(os.path.join(self.directory, 'ur', 'ur5-afc.urdf'), useFixedBase=True, basePosition=[0, 0, 0.5], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.pid)
        control_joint_indices = [0, 1, 2, 3, 4, 5]
        tool_index = 6

        # Reset the joint states
        target_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, -1.57]
        self.reset_joint_states(robot, joint_ids=control_joint_indices,  positions=target_joint_positions)

        # Let the environment settle
        for _ in range(100):
            p.stepSimulation()

        return robot, control_joint_indices, tool_index
    
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
                                   basePosition=position, baseOrientation=orientation)
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


    def get_observation(self):
        # get current end-effector position and velocity in the task/operational space
        x = np.asarray(p.getLinkState(self.robot, self.tool)[0])
        dx = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[6])
        o = np.asarray(p.getLinkState(self.robot, self.tool)[1])
        do = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[7])

        # get end-effector velocity in task space
        tool_velocity = np.concatenate((dx, do))

        # get position error
        position_err = self.target_position - x
        orientation_err = self.target_orientation - o
        pose_error = np.concatenate((position_err, orientation_err))

        # get force error
        F = self.force_torque_sensing(self.robot, joint_ids=6)
        Fz_current = F[2] 
        Fz_error = Fz_current - self.Fz_desired  # record the current error
        force_error = np.array([0, 0, Fz_error, 0, 0, 0])

        return np.concatenate((pose_error, force_error, tool_velocity)).ravel()

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

    def render(self, mode='human', close=False):
        if not self.gui:
            self.gui = True
            p.disconnect(self.pid)
            self.pid = p.connect(p.GUI)
