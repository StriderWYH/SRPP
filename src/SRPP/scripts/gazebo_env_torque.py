#!/home/ur3/anaconda3/envs/spinningup/bin/python3
import sys
import copy
import time
import rospy
import gym
import numpy as np
from project_header import *
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
from std_srvs.srv import Empty
import quaternion

import time
import math

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems


time_delta = 0.1 # we control the robot arm every 0.1s

obs_torque_low_bd = np.array([-50, -50, -50, -50, -12, -12, -12])  # NM as unit
obs_torque_high_bd = np.array([50, 50, 50, 50, 12, 12, 12])
obs_angle_low_bd = np.array([-2.8973,-1.7628,-2.8973 , -3.0718,-2.8973,-0.0175,-2.8973])
obs_angle_high_bd = np.array([2.8973, 1.7628, 2.8973,-0.0698,2.8973,3.7525,2.8973 ])

observation_low_bd = np.hstack((obs_torque_low_bd, obs_angle_low_bd)) # 14 degree
observation_high_bd = np.hstack((obs_torque_high_bd, obs_angle_high_bd))

target_obj = np.array([300,50,150]) # millimeter

class FrankaEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, thr=100, weight=0.00069,vel=4.0,accel=4.0):

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)				#恢复仿真
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)				    #暂停仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)	




        # 定义观测空间, the torque space is the boundaries of the torque
        self.torque_space = spaces.Box(low = obs_torque_low_bd, high=obs_torque_high_bd, dtype=np.float64)

        self.observation_space = spaces.Box(low = observation_low_bd, high= observation_high_bd, dtype=np.float64)
        # the observation space we define contains the following things:
        # end effector position; end effector orientation; end effector linear velocity;  joint state
        # 3 + 4 + 3 + 7 = 17

        # 定义动作空间, where the action is the variation of the torques
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # temporary action torque
        
        # Initialize ROS node
        rospy.init_node('sacnode')
        
        self.limb = PandaArm()

        while (rospy.is_shutdown()):
            print("ROS is shutdown!")
        
        # 初始化环境
        self.reset()
        print("endpose:", self.limb.endpoint_pose(), "\n")
        print("initial self.limb.gravity_comp() + self.limb.coriolis_comp(): ", self.limb.gravity_comp() + self.limb.coriolis_comp(), "\n")
        # print(self.state_orientation)
        self.target_xyz = target_obj    # target xyz position
        self.target_orientation = self.state_orientation
        self.thr = thr                 # hyperparameter, in millimeter
        self.weight = weight              # hyperparameter
        self.current_error = -math.inf
        self.count = 0
        self.count_done = 0
        self.seed()

        self.vel = vel
        self.accel = accel
        self.action_renomal = 7
        self.time_delta = time_delta

    def reset(self, startstate=None):
        
        # 初始化机械臂状态和目标位置
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        self.limb.enable_robot()
        time.sleep(1.0)
        print("reset start")
        self.joint_state = np.array([self.limb._neutral_pose_joints[n] for n in self.limb._joint_names]) # the whole 7 joints
        self.limb.exec_position_cmd(self.joint_state)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        #
        time.sleep(2)
        #
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.torque_state = self.limb.gravity_comp() + self.limb.coriolis_comp()
        self.state_xyz = 1000*self.limb.endpoint_pose()['position']
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is an array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        self.angular_vel = self.limb.endpoint_velocity()['angular']

        # temp = [self.state_xyz[0], self.state_xyz[1], self.state_xyz[2], self.state_orientation.x, self.state_orientation.y,
        #               self.state_orientation.z, self.state_orientation.w, self.linear_vel[0], self.linear_vel[1], self.linear_vel[2]]
        # self.state = np.hstack((temp, self.joint_state))
        self.state = np.hstack((self.torque_state, self.joint_state))
        # 返回初始状态
        # print(self.state)
        # print("\n")
        return self.state

    def update_robot_state(self):
        self.joint_state = np.array([self.limb._joint_angle[n] for n in self.limb._joint_names]) # the whole 7 joints
        self.state_xyz = 1000*self.limb.endpoint_pose()['position']
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is an array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        self.angular_vel = self.limb.endpoint_velocity()['angular']


        # temp = [self.state_xyz[0], self.state_xyz[1], self.state_xyz[2], self.state_orientation.x, self.state_orientation.y,
        #               self.state_orientation.z, self.state_orientation.w, self.linear_vel[0], self.linear_vel[1], self.linear_vel[2]]
        # self.state = np.hstack((temp, self.joint_state))
        self.state = np.hstack((self.torque_state, self.joint_state))

    def step(self, action, move = True):
        # 更新机械臂状态

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        now_torque = self.torque_state + action * self.action_renomal   # renormalize to [-7,+7]
        
        print(now_torque)
        # print("\n")

        # First, check the scope of the new torque
        if self.torque_space.contains(now_torque):
            self.torque_state = now_torque
            print("exec correct torque")
            self.limb.exec_torque_cmd(now_torque)

        else:
            adjust_torque = np.array([0,0,0,0,0,0,0])
            for n in range(0,7):
                if now_torque[n] <= obs_torque_high_bd[n] and now_torque[n] >= obs_torque_low_bd[n]:
                    adjust_torque[n] = now_torque[n]
                elif now_torque[n] > obs_torque_high_bd[n]:
                    adjust_torque[n] = obs_torque_high_bd[n]
                else:
                    adjust_torque[n] = obs_torque_low_bd[n]
            self.torque_state = adjust_torque
            print("exec adjusted torque")
            self.limb.exec_torque_cmd(adjust_torque)


        # Second, implement the new torque
        # 首先unpause，发布消息后，开始仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        #
        time.sleep(self.time_delta)
        #
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.update_robot_state()
        distance = np.linalg.norm(self.state_xyz - self.target_xyz)  # Euclidean distance
        if self.limb.in_safe_state() and self.check_xyz(self.state_xyz) :  # check whether the robot is fine
            # 计算奖励
            reward = self._compute_reward()
            if distance <= self.thr:
                done = True
                reward = 500
                self.count_done += 1
            elif distance < self.current_error:
                reward = 10
                self.current_error = distance
                done = False
            else:
                done = False

            if not self.torque_space.contains(now_torque):
                reward -= 50
        else:
            print("not safe state\n")
            done = True
            reward = -50
            self.reset()      # reinitialize the world

        info = {
            'distance_error': distance,
            'target_position': self.target_xyz,
            'current_position': self.state_xyz,
            'current_obs' : self.state
        }


        # 返回下一个状态、奖励、是否终止以及其他信息
        return self.state, reward, done, info

    def render(self):
        # 可选的渲染方法
        pass


    def _compute_reward(self):
        # 根据末端执行器位置与目标位置之间的距离计算奖励
        distance = np.linalg.norm(self.state_xyz - self.target_xyz) # millimeter
        reward = -distance*self.weight # 使用负距离作为奖励的一部分，目标是最小化距离

        # 加入关节速度奖励部分
        velocity_reward = -0.05 * np.linalg.norm(self.linear_vel - self.vel)  # 使用关节速度的负平方作为奖励的一部分
        reward += velocity_reward

        # we need to give some constraint to the orientation
        delta_ori = self.quatdiff_in_euler(self.state_orientation, self.target_orientation)
        ori_loss = -0.1*np.linalg.norm(delta_ori)
        reward += ori_loss

        # we should try best to not let the torque too big
        torque_loss = -np.linalg.norm(self.torque_state) * self.weight
        reward += torque_loss
        return reward


    def quatdiff_in_euler(self, quat_curr, quat_des):
        """
            Compute difference between quaternions and return 
            Euler angles as difference
        """
        curr_mat = quaternion.as_rotation_matrix(quat_curr)
        des_mat = quaternion.as_rotation_matrix(quat_des)
        rel_mat = des_mat.T.dot(curr_mat)
        rel_quat = quaternion.from_rotation_matrix(rel_mat)
        vec = quaternion.as_float_array(rel_quat)[1:]
        if rel_quat.w < 0.0:
            vec = -vec
            
        return -des_mat.dot(vec)
    
    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    @staticmethod
    def check_xyz(xyz):
        if xyz[2]>=0:
            return True
        else:
            return False
