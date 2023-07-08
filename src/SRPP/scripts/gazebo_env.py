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
import time
import math

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems

# Any other global variable you want to define

# Position for UR3 initialization, in radian
go_away = np.array([180*PI/180.0, -90*PI/180.0, 90*PI/180.0, -90*PI/180.0, -90*PI/180.0, 135*PI/180.0])

# Position for the target object with respect to the world coordinate
target_obj = np.array([300,50,150]) # millimeter

# obervation bounds
obs_low_bd = np.array([-2.8973,-1.7628,-2.8973 , -3.0718,-2.8973,-0.0175,-2.8973])
obs_high_bd = np.array([2.8973, 1.7628, 2.8973,-0.0698,2.8973,3.7525,2.8973 ])


# 20Hz
SPIN_RATE = 20

# UR3 home location
home = np.array([0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0])


thetas = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

digital_in_0 = 0
analog_in_0 = 0.0

suction_on = True
suction_off = False

current_io_0 = False
current_position_set = False

image_shape_define = False

panda_joint6_hm =  1.5411935298621688
panda_joint7_hm = 0.7534486589746342

class gazebo_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作
    # 环境中会用的全局变量可以声明为类（self.）的变量
    def __init__(self, thr=100, weight=0.00069,vel=4.0,accel=4.0):
        self.action_space = spaces.Box(low=-5*PI/180, high=5*PI/180, shape=(5,), dtype=np.float64)  # [just the first 5 joints ]
        self.observation_space = spaces.Box(low=obs_low_bd, high=obs_high_bd, dtype=np.float64)     # the scope for the whole 7 joints

        # Initialize ROS node
        rospy.init_node('sacnode')

        self.limb = PandaArm()
        # Check if ROS is ready for operation
        while (rospy.is_shutdown()):
            print("ROS is shutdown!")
        # set to neutral position
        self.limb.move_to_neutral()

        # initial position
        self.state = np.array([self.limb._neutral_pose_joints[n] for n in self.limb._joint_names]) # the whole 7 joints
        self.state_xyz,_ = self.limb.forward_kinematics(self.state)
        self.target_xyz = target_obj    # target xyz position
        self.thr = thr                 # hyperparameter, in millimeter
        self.weight = weight              # hyperparameter
        self.current_error = -math.inf
        self.count = 0
        self.count_done = 0
        self.seed()

        self.vel = vel
        self.accel = accel






    def step(self, action, move=False):
        # 接收一个动作，执行这个动作
        # 用来处理状态的转换逻辑
        # 返回动作的回报、下一时刻的状态、以及是否结束当前episode及调试信息
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action_seven_joint = np.hstack((action, np.array([panda_joint6_hm,panda_joint7_hm])))
        x = self.state + action_seven_joint


        # 在这里做一下限定，如果下一个动作导致智能体越过了环境边界（即不在状态空间中），则无视这个动作
        pos, ori = self.limb.ee_pose()
        next_state = x
        next_state_xyz,_ = self.limb.forward_kinematics(x)
        if self.check_xyz(next_state_xyz) and self.observation_space.contains(next_state):
            self.state = next_state
            self.state_xyz = next_state_xyz
            # print(next_state)
            # print("\n")
        # When testing, move the arm through ROS
            if move==True:
                self.limb.exec_position_cmd(self.state)

        else:
            self.state = self.state
            self.state_xyz = self.state_xyz

        self.counts += 1

        # 2nd and 3rd return value: reward and done
        # 如果到达了终点，给予一个回报
        # 在复杂环境中多种状态的反馈配比很重要
        distance = np.linalg.norm(self.state_xyz - self.target_xyz)  # Euclidean distance
        if distance <= self.thr:
            reward = 30
            self.count_done += 1
            done = True
        elif distance < self.current_error:
            reward = 10
            self.current_error = distance
            done = False
        else:
            reward = self.get_reward(distance, weight=self.weight)
            done = False

        # 4th return value: info structure
        info = {
            'distance_error': distance,
            'target_position': self.target_xyz,
            'current_position': self.state_xyz,
            'current_thetas' : self.state
        }

        return self.state, reward, done, info

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    # 在训练的时候，可以不指定startstate，随机选择初始状态，以便能尽可能全的采集到的环境中所有状态的数据反馈
    def reset(self, startstate=None):
        if startstate==None:
            self.state = self.observation_space.sample()
        else:   # 在训练完成测试的时候，可以根据需要指定从某个状态开始
            if self.observation_space.contains(startstate):
                self.state = startstate
            else:
                self.state = self.observation_space.sample()
        self.counts = 0
        return self.state

    # render()绘制可视化环境的部分都写在这里
    def render(self, mode='human'):
        # 布置状态

        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def get_reward(distance, weight =0.69):

        return -1*distance*weight
    @staticmethod
    def check_xyz(xyz):
        if xyz[2]>=0:
            return True
        else:
            return False