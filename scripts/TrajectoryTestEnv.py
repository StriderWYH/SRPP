#!/usr/bin/python3
import sys
import copy

import gymnasium
import rospy
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
# from gymnasium.envs.classic_control import rendering
from std_srvs.srv import Empty
import quaternion

import time
import math
import random

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems

time_delta = 0.08  # we control the robot arm every 0.1s

obs_torque_low_bd = np.array([-50, -50, -50, -50, -12, -12, -12])  # NM as unit
obs_torque_high_bd = np.array([50, 50, 50, 50, 12, 12, 12])
obs_angle_low_bd = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
obs_angle_high_bd = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

target_obj = np.array([300, 50, 150])  # millimeter
PI = np.pi

kl_05 = np.log(2 + np.sqrt(3))  # let A(error_mean) to be 0.5


class ErrorBuffer:
    """
    A simple FIFO experience replay buffer for error values.
    """

    def __init__(self, zone_size=200, zone_num=8, frag_point_num=0.8):
        self.zone_size = zone_size  # capacity of a zone
        self.zone_num = zone_num  # how many zone set
        self.frag_point_num = frag_point_num    # number of points per zone
        assert isinstance(self.frag_point_num, int), "fragment length must be an integer"
        # initialize the error buffer
        self.d_error_buf = np.zeros([self.zone_num, self.zone_size], dtype=np.float32)
        self.d_error_buf.fill(99)
        self.j_error_buf = np.zeros([self.zone_num, self.zone_size], dtype=np.float32)
        self.j_error_buf.fill(99)
        self.ori_error_buf = np.zeros([self.zone_num, self.zone_size], dtype=np.float32)
        self.ori_error_buf.fill(99)
        self.v_error_buf = np.zeros([self.zone_num, self.zone_size], dtype=np.float32)
        self.v_error_buf.fill(99)

        self.ptr, self.size, self.max_size = 0, 0, self.zone_size
        self.reach_full = np.zeros([self.zone_num, 1], dtype=np.float32)
        self.reach_full.fill(False)
        self.count = 0

    def store(self, ord, d_error, j_error, ori_error, v_error):
        zone_ord = ord // self.frag_point_num
        assert 0 <= zone_ord < self.zone_num, "zone ord exceed the range"
        self.d_error_buf[zone_ord, self.ptr] = d_error
        self.j_error_buf[zone_ord, self.ptr] = j_error
        self.ori_error_buf[zone_ord, self.ptr] = ori_error
        self.v_error_buf[zone_ord, self.ptr] = v_error
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def error_step(self):
        self.count += 1

    def mean_error(self, ord):
        zone_ord = ord // self.frag_point_num
        d_mean = self.d_error_buf[zone_ord].mean()
        j_mean = self.j_error_buf[zone_ord].mean()
        ori_mean = self.ori_error_buf[zone_ord].mean()
        v_mean = self.v_error_buf[zone_ord].mean()
        return d_mean, j_mean, ori_mean, v_mean

    def at_brim(self, ord):
        zone_ord = ord // self.frag_point_num
        return self.reach_full[zone_ord]

    def flag_reset(self, ord):
        zone_ord = ord // self.frag_point_num
        self.reach_full[zone_ord] = False

    def flag_set(self):
        self.reach_full.fill(True)


class CircleTrajectoryTest(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # thr_dist(mm), thr_jt(rad), thr_ori(), thr_vel(m/s)
    def __init__(self, thr_dist=10, thr_jt=0.2 * PI, thr_ori=0.08, thr_vel=0.01, weight=0.00069, vel=4.0, accel=4.0,
                 length=159, update_k_everysteps=40000, zone_num=10):

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 恢复仿真
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        # 定义观测空间
        self.torque_space = spaces.Box(low=obs_torque_low_bd, high=obs_torque_high_bd, dtype=np.float64)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float64)
        # 定义动作空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # temporary action torque

        # Initialize ROS node
        rospy.init_node('sacnode')

        self.limb = PandaArm()
        while (rospy.is_shutdown()):
            print("ROS is shutdown!")
        self.target_xyz = None
        self.target_orientation = None
        self.state_orientation = None
        self.state_xyz = None
        self.last_xyz = None
        self.freeze_step = 0
        self.ori_loss = 0
        self.state = None
        self.joint_state_error = None
        self.pre_joint_state = None
        self.pre_torque_state = None
        self.init_joint_state = None
        self.joint_state = None
        self.startpoint = None
        self.freeze_pos_step = None
        self.rms_error = None
        self.time = None
        self.target_velocity = None
        self.ck_point_num = None
        self.sampleAngle = None
        self.samplePoint = None
        self.target_joint_state = None
        self.linear_vel = None
        self.torque_state = None
        self.count_checkpoint = 0
        self.count_done = 0
        self.seed()
        self.vel = vel
        self.accel = accel
        self.traj_length = length
        self.thr = np.array([thr_dist, thr_jt, thr_ori, thr_vel])  # hyperparameter
        self.weight = weight  # hyperparameter

        # # initialize the kernel parameter and the error buffer
        # error_buf_zone_size = 200
        # self.zone_number = zone_num
        # assert (length+1) % self.zone_number == 0, "length+1 must be the multiple of the zone number"
        # frag_point_num = int((length+1) / self.zone_number)
        # self.error_buf = ErrorBuffer(zone_size=error_buf_zone_size, zone_num= self.zone_number, frag_point_num=frag_point_num)
        # # self.l_d, self.l_j, self.l_ori, self.l_v = 3.6, 0.8, 3, 3  # initial value of kernel parameter
        # # self.l_d, self.l_j, self.l_ori, self.l_v = 3.6, 2.8, 3, 3  # 25 epoch(4000stp/ep) value of kernel parameter
        # # self.l_d, self.l_j, self.l_ori, self.l_v = 20.886, 2.8, 3.45, 3 # 48 epoch (192k) value of kernel parameter
        # # self.l_d, self.l_j, self.l_ori, self.l_v = 83.6042, 3.656347, 4.482546, 3.365036  # 124 epoch (496k) value of kernel parameter
        # self.l_d, self.l_j, self.l_ori, self.l_v = None, None, None, None
        # self.l_init(self.zone_number)


        ##################################################################
        # 初始化环境
        self.workSpaceInit(length=length)
        self.reset()

    def l_init(self,zone_number):
        self.l_d = np.zeros([zone_number, ], dtype=np.float32)
        self.l_d.fill(3.6)
        self.l_j = np.zeros([zone_number, ], dtype=np.float32)
        self.l_j.fill(0.8)
        self.l_ori = np.zeros([zone_number, ], dtype=np.float32)
        self.l_ori.fill(3)
        self.l_v = np.zeros([zone_number, ], dtype=np.float32)
        self.l_v.fill(3)
        for i in range(0, 3):
            self.l_d[i] = 83.6042
            self.l_j[i] = 3.656347
            self.l_ori[i] = 4.536543766
            self.l_v[i] = 3.365036
        self.l_d[3] = 20.886
        self.l_j[3] = 2.8
        self.l_ori[3] = 3.45
        self.l_v[3] = 3.365036

    def workSpaceInit(self, length, delta_y=5):
        """
          The workspace of the robot arm is 855mm
          return the tuple (sample Point, sample angle), each has two index, one for start, one for target
        """
        joint_state = np.array(
            [self.limb._neutral_pose_joints[n] for n in self.limb._joint_names])  # the whole 7 joints
        print("joint state is :", joint_state)
        self.limb.exec_position_cmd(joint_state)  # move to the neutral position
        time.sleep(2.0)
        self.target_orientation = self.limb.endpoint_pose()['orientation']  # this is am array of type quaternion
        print("target orientation is :", self.target_orientation, "\n")

        self.samplePoint = []  # millimeter
        self.sampleAngle = []
        self.ck_point_num = length
        for t in range(0, length + 1):
            r = 300
            x = r * np.cos(0.0125 * PI * t) + r + 100
            y = r * np.sin(0.025 * PI * t)
            self.samplePoint.append(np.array([x, y, 400]))
            temp_check = self.limb.inverse_kinematics(self.samplePoint[t] / 1000,
                                                      self.target_orientation)
            assert temp_check[0], "position t has no inverse kinematics"
            self.sampleAngle.append(temp_check[1])
        # print("\n show time: \n")
        # for t in range(0, length+1):
        #     self.limb.exec_position_cmd(self.sampleAngle[t])  # move to the neutral position
        #     time.sleep(0.5)
        # joint_state = np.array(
        #     [self.limb._neutral_pose_joints[n] for n in self.limb._joint_names])  # the whole 7 joints
        # self.limb.exec_position_cmd(joint_state)  # move to the neutral position
        # time.sleep(2.0)
        return

    def getSelectedPointInLine(self, order=0):
        # choice = random.choice(range(0,length))
        PointPair = []
        AnglePair = []
        PointPair.append(self.samplePoint[order])
        AnglePair.append(self.sampleAngle[order])
        PointPair.append(self.samplePoint[(order + 1)])
        AnglePair.append(self.sampleAngle[(order + 1)])
        return PointPair, AnglePair

    def reset(self, seed=None, options=None, startstate=None):
        self.freeze_step = 0
        self.freeze_pos_step = 0
        # print("last turn's rms_error:", self.rms_error/self.time)
        self.rms_error = 0
        print("checkpoint this round is:", self.count_checkpoint)
        # print("count done is:", self.count_done, "\n")
        self.count_checkpoint = 0
        # print("former time is", time)
        self.time = 0
        # 初始化机械臂状态和目标位置
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        self.limb.enable_robot()
        # time.sleep(1.0)
        # fetch the sample point pair
        sample = self.getSelectedPointInLine()
        # print("sample is :", sample)
        self.startpoint = sample[0][0]
        self.joint_state = sample[1][0]
        self.target_xyz = sample[0][1]
        self.target_joint_state = sample[1][1]
        self.init_joint_state = self.joint_state
        self.pre_joint_state = self.joint_state
        self.joint_state_error = self.target_joint_state - self.joint_state
        # print("init_joint_state is :", self.init_joint_state)
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # move to the start point
        self.limb.exec_position_cmd(self.joint_state)
        time.sleep(2.0)
        self.torque_state = np.array(
            [self.limb.joint_efforts()[n] for n in self.limb._joint_names])  # the whole 7 joints
        self.state_xyz = 1000 * self.limb.endpoint_pose()['position']  # millimeter
        self.last_xyz = self.state_xyz
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is a array of type quaternion
        # print("orientation is :\n", self.state_orientation)
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        # print("linear velocity is:", self.linear_vel)
        self.target_velocity = 0.001 * (self.target_xyz - self.state_xyz) / time_delta
        self.ori_loss = 0
        self.pre_torque_state = self.torque_state
        temp = [
            self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
            self.state_orientation.x, self.state_orientation.y,
            self.state_orientation.z, self.state_orientation.w
        ]
        # print("torque_state is:", self.torque_state/50)
        # print("pre_joint_state is:", self.pre_joint_state / PI)
        # print("target_joint_state is:", self.target_joint_state/PI)
        alter_torque_state = self.Compute_torque_state(self.torque_state)
        alter_pre_torque_state = self.Compute_torque_state(self.pre_torque_state)
        state = np.hstack((self.joint_state / PI, self.target_joint_state / PI, self.joint_state_error / PI,
                           alter_torque_state, alter_pre_torque_state,
                           self.state_xyz / 1000, self.target_xyz / 1000))
        self.state = np.hstack((state, temp))
        # 返回初始状态
        # print(self.state)
        # print("\n")
        return self.state, {}

    def update_robot_state(self):

        self.last_xyz = self.state_xyz
        # self.pre_joint_state_t2 = self.pre_joint_state
        self.pre_joint_state = self.joint_state
        self.joint_state = np.array([self.limb._joint_angle[n] for n in self.limb._joint_names])  # the whole 7 joints
        # print("joint_state is : \n",self.joint_state)
        self.state_xyz = 1000 * self.limb.endpoint_pose()['position']
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is an array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        # print("Linear velocity is: ", self.linear_vel, "\n")
        sample = self.trajectory()
        # print("sample is :", sample)
        self.target_xyz = sample[0]
        self.target_joint_state = sample[1]
        self.joint_state_error = self.target_joint_state - self.joint_state
        alter_torque_state = self.Compute_torque_state(self.torque_state)
        alter_pre_torque_state = self.Compute_torque_state(self.pre_torque_state)
        # print("alter_torque_state is:", alter_torque_state)
        temp = [
            self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
            self.state_orientation.x, self.state_orientation.y,
            self.state_orientation.z, self.state_orientation.w
        ]
        state = np.hstack((self.joint_state / PI, self.target_joint_state / PI, self.joint_state_error / PI,
                           alter_torque_state, alter_pre_torque_state,
                           self.state_xyz / 1000, self.target_xyz / 1000))
        self.state = np.hstack((state, temp))

    def step(self, action, move=True):

        self.time += 1
        time_d = self.time
        reward = 0
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        temp_torque = self.Compute_torque_action(action)
        # print("temp_torque is:", temp_torque)
        now_torque = self.torque_state + temp_torque  # renormalize
        # print("now torque_state is:", now_torque)

        if self.torque_space.contains(now_torque):
            self.freeze_step = 0
            self.pre_torque_state = self.torque_state
            self.torque_state = now_torque
            # print("exec torque")
            self.limb.exec_torque_cmd(now_torque)
        else:
            self.freeze_step += 1
            if self.freeze_step >= 10:
                reward -= 5
                print("freeze step reset \n")
                return self.state, reward, True, False, {}
            self.pre_torque_state = self.torque_state
            self.limb.exec_torque_cmd(self.torque_state)
            reward -= 5

        # 首先unpause，发布消息后，开始仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # 接下来需要停一小段时间，让小车以目前的速度行驶一段时间，TIME_DELTA = 0.1s
        time.sleep(time_delta)
        # 然后我们停止仿真，开始观测目前的状态
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # update  the robot state
        self.update_robot_state()

        # 计算奖励
        reward_delta = self._compute_reward()
        reward += reward_delta

        joint_state_difs = self.joint_state - self.target_joint_state
        # print("joint_state_difs are:", joint_state_difs)
        dif_count = 0
        for dif in joint_state_difs:
            dif_count += 1
            if abs(dif) > 0.75 * PI and dif_count <= 6:  # the last joint is not important
                # print(self.joint_state_error)
                reward_m = (self.traj_length-self.time)
                reward -= reward_m
                # reward_m = (self.traj_length - self.time)
                # reward -= reward_m
                print("dif torque reset \n")
                print("reward is :", reward, "\n")
                return self.state, reward, True, False, {}

        distance = 0
        if self.limb.in_safe_state() and self.check_xyz(self.state_xyz):  # check whether the robot is fine
            distance = np.linalg.norm(self.state_xyz - self.target_xyz)  # Euclidean distance
            joint_error = np.linalg.norm(self.joint_state_error[0:5], ord=1)
            if distance <= self.thr[0] and joint_error <= self.thr[1]:
                self.count_checkpoint += 1
                if time_d >= self.ck_point_num:
                    done = True
                    reward += 10 + self.count_checkpoint
                    self.count_done += 1
                else:
                    done = False
                    reward += 1
            else:
                if time_d < self.ck_point_num:
                    done = False
                else:
                    reward -= 1
                    done = True
        else:
            print("not safe state\n")
            done = True
            reward -= -5
            # reinitialize the world

        info = {
            'distance_error': distance,
            'target_position': self.target_xyz,
            'current_position': self.state_xyz,
            'current_obs': self.state,
            'rms_error': self.rms_error
        }
        print("reward in sum of this step is :", reward, "\n")
        # 返回下一个状态、奖励、是否终止以及其他信息
        return self.state, reward, done, False, info

    def render(self):
        # 可选的渲染方法
        pass

    def _compute_reward(self, w_dist=0.4, w_joint=0.3, w_ori=0.2, w_vel=0.1):
        reward = 0

        # 根据末端执行器位置与目标位置之间的距离计算惩罚
        d_error = np.linalg.norm((self.state_xyz - self.target_xyz)/1000)

        self.rms_error += (d_error/1000) ** 2

        reward -= d_error

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

    def trajectory(self, backtrace=0):
        time_d = self.time
        if time_d <= self.ck_point_num:
            time = time_d - backtrace
        else:
            time = self.ck_point_num - backtrace
        return (self.samplePoint[time], self.sampleAngle[time])  # Attention here! This is taken in millimeter

    def Compute_torque_state(self, torque):
        torque_1234 = np.array([torque[0], torque[1], torque[2], torque[3]]) / 50
        torque_567 = np.array([torque[4], torque[5], torque[6]]) / 10

        return np.hstack((torque_1234, torque_567))

    def Compute_torque_action(self, torque):
        torque_1234 = np.array([torque[0], torque[1], torque[2], torque[3]]) * 5
        torque_567 = np.array([torque[4], torque[5], torque[6]])

        return np.hstack((torque_1234, torque_567))

    @staticmethod
    def check_xyz(xyz):
        if xyz[2] >= 0:
            return True
        else:
            return False
