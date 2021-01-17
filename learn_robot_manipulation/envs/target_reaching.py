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

from .world_creation import WorldCreation
from .utils import Utils

class TargetReachingEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, robot_type = 'rrbot', action_robot_len=2, obs_robot_len=16, reward_type=2, time_step = 0.02):

        """
        Initialize the environment.

        Args:
            action_robot_len (int, None): length of action space
            obs_robot_len (int, None): length of observation space
        """

        # Start the bullet physics server
        self.gui = False
        self.id = p.connect(p.DIRECT)
        
        self.robot_type = robot_type
        # self.max_speed = np.pi #rad/sec
        self.max_speed = 1.0 #rad/sec
        self.time_step = time_step
        self.reward_ver = reward_type
        
        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len
        self.action_space = spaces.Box(low=np.float32(np.array([-self.max_speed]*self.action_robot_len)), high=np.float32(np.array([self.max_speed]*self.action_robot_len)), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.float32(np.array([-1.0]*self.obs_robot_len)), high=np.float32(np.array([1.0]*self.obs_robot_len)), dtype=np.float32)

        self.target_position = np.array([0.0, 0.0, 0.0]) #[x,y,z]
        self.obs_state = None
        self.target_radius = 0.2

        #attributes
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.setup_timing()
        self.seed(1001) #initialize a seed

        self.world_creation = WorldCreation(self.id, robot_type=self.robot_type, time_step=self.time_step, np_random=self.np_random)
        self.utils = Utils(self.id, self.np_random)

        self.width = 1920
        self.height = 1080
        # self.width = 3840
        # self.height = 2160
        # self.width = 400
        # self.height = 300
    
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


        self.iteration +=1
        if self.last_sim_time is None:
            self.last_sim_time = time.time()

        p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=action,
                                    physicsClientId=self.id)

        # Update robot position
        p.stepSimulation(physicsClientId=self.id)
        if self.gui:
            # Slow down time so that the simulation matches real time
            self.slow_time()

        #get an observation of state
        obs_state = self._get_obs()

        # get reward        
        distance_to_target = self.utils.euclidian_distance(self.target_position, self.utils.tool_position(self.robot, self.tool))
        # print('distance-to-target %.2f:' % (distance_to_target))

        reward_dist = -distance_to_target
        reward_ctrl = -np.square(action).sum()
        
        reached = False
        terminated = False
        if distance_to_target < self.target_radius:
            # print('Reached Goal !!')
            reached = True

        # Reward: 0
        if self.reward_ver == 0:
            reward = reward_dist
        # Reward: 1
        if self.reward_ver == 1:
            reward = reward_dist + reward_ctrl
        # # Reward: 2 or 3
        if self.reward_ver == 2:
            if not reached:
                reward = - 1.0 + reward_ctrl
                # reward = - 1.0
            else:
                reward = 1000
               
        # if (self.iteration*self.dt) > self.max_time:
        #     print('Time Limit Reached !!')
        #     terminated = True

        # done = reached or terminated
        done = False
        
        info = {'Position Error': distance_to_target, 'task_success': int(reached)}
        # print(info)

        return obs_state, reward, done, info

    def _get_obs(self):
        # get current end-effector position and velocity in the task/operational space
        x = np.asarray(p.getLinkState(self.robot, self.tool)[0])
        dx = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[6])
        o = np.asarray(p.getLinkState(self.robot, self.tool)[1])
        do = np.asarray(p.getLinkState(self.robot, self.tool, computeLinkVelocity=1)[7])

        # get end-effector velocity in task space
        tool_pose = np.concatenate((x, o))
        tool_velocity = np.concatenate((dx, do))

        return np.concatenate((self.target_position, tool_pose, tool_velocity)).ravel()

    def reset(self):
        # print('Reset Env!!')

        self.setup_timing()
        self.robot, self.controllable_joints, self.tool, self.joint_lower_limits, self.joint_upper_limits, self.target, self.target_position, self.target_orientation = self.world_creation.create_new_world()

        # self.create_world()
        # self.robot, self.controllable_joints, self.tool, self.joint_lower_limits, self.joint_upper_limits  = self.load_robot()
        # self.target, self.targte_position, self.target_orientation = self.load_target()
        
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # p.setGravity(0, 0, -1, physicsClientId=self.id)

        return self._get_obs()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0    

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            p.disconnect(self.id)
            # self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0')
            # self.id = p.connect(p.GUI)            
            self.world_creation = WorldCreation(self.id, robot_type=self.robot_type, time_step=self.time_step, np_random=self.np_random)
            # self.util = Util(self.id, self.np_random)
            # # print('Physics server ID:', self.id)
