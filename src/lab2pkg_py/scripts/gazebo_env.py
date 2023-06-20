#!/home/ur3/anaconda3/envs/spinningup/bin/python3
import sys
import copy
import time
import rospy
import gym
import numpy as np
from project_header import *
from project_func import *
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import time
import math


# Any other global variable you want to define

# Position for UR3 initialization, in radian
go_away = np.array([270*PI/180.0, -90*PI/180.0, 90*PI/180.0, -90*PI/180.0, -90*PI/180.0, 135*PI/180.0])

# Position for the target object with respect to the base?
target_obj = np.array([400, 250,100]) # millimeter

# obervation bounds
obs_low_bd = np.array([0.5*PI,-PI,-5*PI/180, -185*PI/180])
obs_high_bd = np.array([3*PI/2, 0, 140*PI/180,-PI/2])


# 20Hz
SPIN_RATE = 20

# UR3 home location
home = np.array([0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0, 0*PI/180.0])

# UR3 current position, using home position for initialization
current_position = copy.deepcopy(home)

thetas = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

digital_in_0 = 0
analog_in_0 = 0.0

suction_on = True
suction_off = False

current_io_0 = False
current_position_set = False

image_shape_define = False


"""
Whenever ur3/gripper_input publishes info this callback function is called.
"""
def input_callback(msg):

    global digital_in_0
    digital_in_0 = msg.DIGIN
    digital_in_0 = digital_in_0 & 1 # Only look at least significant bit, meaning index 0


"""
Whenever ur3/position publishes info, this callback function is called.
"""
def position_callback(msg):

    global thetas
    global current_position
    global current_position_set

    thetas[0] = msg.position[0]
    thetas[1] = msg.position[1]
    thetas[2] = msg.position[2]
    thetas[3] = msg.position[3]
    thetas[4] = msg.position[4]
    thetas[5] = msg.position[5]

    current_position[0] = thetas[0]
    current_position[1] = thetas[1]
    current_position[2] = thetas[2]
    current_position[3] = thetas[3]
    current_position[4] = thetas[4]
    current_position[5] = thetas[5]

    current_position_set = True


"""
Function to control the suction cup on/off
"""
def gripper(pub_cmd, loop_rate, io_0):

    global SPIN_RATE
    global thetas
    global current_io_0
    global current_position

    error = 0
    spin_count = 0
    at_goal = 0

    current_io_0 = io_0

    driver_msg = command()
    driver_msg.destination = current_position
    driver_msg.v = 1.0
    driver_msg.a = 1.0
    driver_msg.io_0 = io_0
    pub_cmd.publish(driver_msg)

    while(at_goal == 0):

        if( abs(thetas[0]-driver_msg.destination[0]) < 0.0005 and \
            abs(thetas[1]-driver_msg.destination[1]) < 0.0005 and \
            abs(thetas[2]-driver_msg.destination[2]) < 0.0005 and \
            abs(thetas[3]-driver_msg.destination[3]) < 0.0005 and \
            abs(thetas[4]-driver_msg.destination[4]) < 0.0005 and \
            abs(thetas[5]-driver_msg.destination[5]) < 0.0005 ):

            #rospy.loginfo("Goal is reached!")
            at_goal = 1

        loop_rate.sleep()

        if(spin_count >  SPIN_RATE*5):

            pub_cmd.publish(driver_msg)
            rospy.loginfo("Just published again driver_msg")
            spin_count = 0

        spin_count = spin_count + 1

    return error


"""
Move robot arm from one position to another
"""
def move_arm(pub_cmd, loop_rate, dest, vel, accel):

 # dest are the six angles of the arm
    global thetas
    global SPIN_RATE

    error = 0
    spin_count = 0
    at_goal = 0

    driver_msg = command()
    driver_msg.destination = dest
    driver_msg.v = vel
    driver_msg.a = accel
    driver_msg.io_0 = current_io_0
    pub_cmd.publish(driver_msg)

    loop_rate.sleep()

    while(at_goal == 0):

        if( abs(thetas[0]-driver_msg.destination[0]) < 0.0005 and \
            abs(thetas[1]-driver_msg.destination[1]) < 0.0005 and \
            abs(thetas[2]-driver_msg.destination[2]) < 0.0005 and \
            abs(thetas[3]-driver_msg.destination[3]) < 0.0005 and \
            abs(thetas[4]-driver_msg.destination[4]) < 0.0005 and \
            abs(thetas[5]-driver_msg.destination[5]) < 0.0005 ):

            at_goal = 1
            #rospy.loginfo("Goal is reached!")

        loop_rate.sleep()

        if(spin_count >  SPIN_RATE*5):

            pub_cmd.publish(driver_msg)
            rospy.loginfo("Just published again driver_msg")
            spin_count = 0

        spin_count = spin_count + 1

    return error

################ Pre-defined parameters and functions no need to change above ################


def move_block(pub_cmd, loop_rate, start_xw_yw_zw, target_xw_yw_zw, vel, accel):

    """
    start_xw_yw_zw: where to pick up a block in global coordinates
    target_xw_yw_zw: where to place the block in global coordinates

    hint: you will use lab_invk(), gripper(), move_arm() functions to
    pick and place a block

    """


    # global variable1
    # global variable2
    error = 0

    rospy.loginfo("Finding the block...")
    # move the arm to grip the block
    move_arm(pub_cmd, loop_rate, start_xw_yw_zw, 4.0, 4.0)
    time.sleep(0.5)
    gripper(pub_cmd,loop_rate,suction_on)
    time.sleep(1.0)
    if not digital_in_0:
        error = 1
        gripper(pub_cmd,loop_rate,suction_off)
        rospy.loginfo("Fail to grip the block")
        return error
    #add in lab5
    move_arm(pub_cmd, loop_rate, mid_angle, 4.0, 4.0)
    rospy.loginfo("Moving to the current destination...")
    #move_arm(pub_cmd,loop_rate,midposition,4.0,4.0)
    # move the are to the destination
    move_arm(pub_cmd,loop_rate,target_xw_yw_zw,4.0,4.0)
    time.sleep(0.5)
    gripper(pub_cmd,loop_rate,suction_off)
    time.sleep(1.0)



    return error


class gazebo_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作
    # 环境中会用的全局变量可以声明为类（self.）的变量
    def __init__(self, thr=10, weight=0.069,vel=4.0,accel=4.0):
        self.action_space = spaces.Box(low=-5*PI/180, high=5*PI/180, shape=(4,), dtype=np.float64)  # [theta1,2,3,4(except for the ending effector)]
        self.observation_space = spaces.Box(low=obs_low_bd, high=obs_high_bd, dtype=np.float64)     # the scope for the first four angle, except for the ending effector
        self.state = go_away   # initial position
        self.state_xyz = None
        self.target_xyz = target_obj    # target xyz position
        self.thr = thr                 # hyperparameter, in millimeter
        self.weight = weight              # hyperparameter
        self.current_error = -math.inf
        self.count = 0
        self.seed()
        self.viewer = rendering.Viewer(520, 200)    # 初始化一张画布
        self.vel = vel
        self.accel = accel

        # Initialize ROS node
        rospy.init_node('sacnode')

        # Initialize publisher for ur3/command with buffer size of 10
        pub_command = rospy.Publisher('ur3/command', command, queue_size=10)

        # Initialize subscriber to ur3/position & ur3/gripper_input and callback fuction
        # each time data is published
        sub_position = rospy.Subscriber('ur3/position', position, position_callback)
        sub_input = rospy.Subscriber('ur3/gripper_input', gripper_input, input_callback)

        # Check if ROS is ready for operation
        while (rospy.is_shutdown()):
            print("ROS is shutdown!")

        # Initialize the rate to publish to ur3/command
        loop_rate = rospy.Rate(SPIN_RATE)

        # set the arm to home position

        move_arm(pub_command, loop_rate, go_away, self.vel, self.accel)

    def step(self, action):
        # 接收一个动作，执行这个动作
        # 用来处理状态的转换逻辑
        # 返回动作的回报、下一时刻的状态、以及是否结束当前episode及调试信息
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x = self.state + action

        # 在这里做一下限定，如果下一个动作导致智能体越过了环境边界（即不在状态空间中），则无视这个动作
        next_state = x
        next_state_xyz = lab_fk(x[0],x[1],x[2],x[3],0,135*PI/180)
        if self.check_xyz(next_state_xyz) and self.observation_space.contains(next_state):
            self.state = next_state
            self.state_xyz = next_state_xyz
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
            done = True
        elif distance < self.current_error:
            reward = 1
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
        for i in range(self.observation_space.n):
            self.viewer.draw_line(
                (20, 30), (100, 30), color=(0, 0, 0)
            ).add_attr(rendering.Transform((100 * i, 0)))

        # 目标位置
        self.viewer.draw_line(
            (20, 30),
            (100, 30),
            color=(0, 1, 0),
        ).add_attr(rendering.Transform((100 * 4, 0)))

        # 绘制当前位置
        self.viewer.draw_polygon(
            [(60, 30), (80, 100), (40, 100)], color=(0, 1, 0)
        ).add_attr(rendering.Transform((100 * self.state, 0)))
        self.viewer.draw_circle(
            20, 20, True, color=(0, 1, 0)
        ).add_attr(rendering.Transform((60 + 100 * self.state, 120)))

        return self.viewer.render(return_rgb_array=mode == 'human')

    def close(self):
        if self.viewer:
            self.viewer.close()

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