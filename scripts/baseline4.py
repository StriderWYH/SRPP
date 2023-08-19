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

    def __init__(self, size):
        self.d_error_buf = np.zeros(size, dtype=np.float32)
        self.j_error__buf = np.zeros(size, dtype=np.float32)
        self.ori_error_buf = np.zeros(size, dtype=np.float32)
        self.v_error_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.reach_full = False

    def store(self, d_error, j_error, ori_error, v_error):
        self.d_error_buf[self.ptr] = d_error
        self.j_error__buf[self.ptr] = j_error
        self.ori_error_buf[self.ptr] = ori_error
        self.v_error_buf[self.ptr] = v_error
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        if self.ptr == self.max_size - 1:
            self.reach_full = True

    def mean_error(self):
        d_mean = self.d_error_buf.mean()
        j_mean = self.j_error__buf.mean()
        ori_mean = self.ori_error_buf.mean()
        v_mean = self.v_error_buf.mean()
        return d_mean, j_mean, ori_mean, v_mean

    def at_brim(self):
        return self.reach_full

    def flag_reset(self):
        self.reach_full = False


class Line4(gymnasium.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # thr_dist(mm), thr_jt(rad), thr_ori(), thr_vel(m/s)
    def __init__(self, thr_dist=10, thr_jt=0.2 * PI, thr_ori=0.08, thr_vel=0.01, weight=0.00069, vel=4.0, accel=4.0,
                 length=160):

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 恢复仿真
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        # 定义观测空间
        self.freeze_step = 0
        self.torque_space = spaces.Box(low=obs_torque_low_bd, high=obs_torque_high_bd, dtype=np.float64)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float64)
        # 定义动作空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # temporary action torque

        # Initialize ROS node
        rospy.init_node('sacnode')

        self.limb = PandaArm()

        while (rospy.is_shutdown()):
            print("ROS is shutdown!")

        # move to the neutral pose
        joint_state = np.array(
            [self.limb._neutral_pose_joints[n] for n in self.limb._joint_names])  # the whole 7 joints
        print("joint state is :", joint_state)
        self.limb.exec_position_cmd(joint_state)  # move to the neutral position
        time.sleep(2.0)
        self.target_orientation = self.limb.endpoint_pose()['orientation']  # this is am array of type quaternion
        print("target orientation is :", self.target_orientation, "\n")
        # alter_joint_state = joint_state + np.array([0,0,0,0,0,0,0.5*PI])
        # print("alterjoint state is :", alter_joint_state)
        # self.limb.exec_position_cmd(alter_joint_state)
        # time.sleep(2.0)
        # test_orientation = self.limb.endpoint_pose()['orientation']
        # print("test orientation is :", test_orientation, "\n")
        self.ori_loss = 0
        ##################################################################
        self.workSpaceInit(length=length)
        joint_state = np.array(
            [self.limb._neutral_pose_joints[n] for n in self.limb._joint_names])  # the whole 7 joints
        self.limb.exec_position_cmd(joint_state)  # move to the neutral position
        time.sleep(2.0)
        self.count_checkpoint = 0
        self.count_done = 0
        # 初始化环境
        self.reset()
        error_buf_size = length * 90
        self.error_buf = ErrorBuffer(error_buf_size)
        # print(self.state_orientation)
        self.target_xyz = target_obj  # target xyz position
        self.thr = np.array([thr_dist, thr_jt, thr_ori, thr_vel])  # hyperparameter
        self.weight = weight  # hyperparameter

        self.count = 0

        self.seed()

        self.vel = vel
        self.accel = accel
        self.traj_length = length

    ### The workspace of the robot arm is 855mm
    # return the tuple (sample Point, sample angle), each has two index, one for start, one for target
    def workSpaceInit(self, length=60, delta_y = 5):
        self.samplePoint = []  # millimeter
        self.sampleAngle = []
        self.ck_point_num = length
        self.target_velocity = np.array([0,0.001*delta_y/(time_delta),0]) # m/s
        for t in range(0, length+1):
            self.samplePoint.append(np.array([500, -390 + delta_y * t, 400]))
            temp_check = self.limb.inverse_kinematics(self.samplePoint[t] / 1000,
                                                      self.target_orientation)
            assert temp_check[0], "position t has no inverse kinematics"
            self.sampleAngle.append(temp_check[1])
        # print("\n show time: \n")
        # for t in range(0, length+1):
        #     self.limb.exec_position_cmd(self.sampleAngle[t])  # move to the neutral position
        #     time.sleep(0.5)
        return

    def getSelectedPointInLine(self,order=0):
        # choice = random.choice(range(0,length))
        PointPair = []
        AnglePair = []
        PointPair.append(self.samplePoint[order])
        AnglePair.append(self.sampleAngle[order])
        PointPair.append(self.samplePoint[(order + 1)])
        AnglePair.append(self.sampleAngle[(order + 1)])
        return (PointPair, AnglePair)

    def reset(self, seed=None, options=None,startstate=None):
        self.freeze_step = 0
        self.freeze_pos_step = 0
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
        print("linear velocity is:", self.linear_vel)
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
        state = np.hstack((self.joint_state/PI,self.target_joint_state/PI,self.joint_state_error/PI,
                           alter_torque_state,alter_pre_torque_state,
                           self.state_xyz/1000, self.target_xyz/1000))
        self.state = np.hstack((state, temp))
        # 返回初始状态
        # print(self.state)
        # print("\n")
        return self.state,{}

    def update_robot_state(self):

        self.last_xyz = self.state_xyz
        # self.pre_joint_state_t2 = self.pre_joint_state
        self.pre_joint_state = self.joint_state
        self.joint_state = np.array([self.limb._joint_angle[n] for n in self.limb._joint_names])  # the whole 7 joints
        # print("joint_state is : \n",self.joint_state)
        self.state_xyz = 1000 * self.limb.endpoint_pose()['position']
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is an array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        print("Linear velocity is: ", self.linear_vel,"\n")
        sample = self.trajectory()
        # print("sample is :", sample)
        self.target_xyz = sample[0]
        self.target_joint_state = sample[1]
        self.joint_state_error = self.target_joint_state - self.joint_state
        alter_torque_state = self.Compute_torque_state(self.torque_state)
        alter_pre_torque_state = self.Compute_torque_state(self.pre_torque_state)
        # print("alter_torque_state is:", alter_torque_state)
        temp = [
            self.linear_vel[0] , self.linear_vel[1] , self.linear_vel[2] ,
            self.state_orientation.x, self.state_orientation.y,
            self.state_orientation.z, self.state_orientation.w
        ]
        state = np.hstack((self.joint_state/PI, self.target_joint_state/PI, self.joint_state_error/PI,
                           alter_torque_state,alter_pre_torque_state,
                           self.state_xyz/1000, self.target_xyz/1000))
        self.state = np.hstack((state, temp))

    def step(self, action, move=True):

        self.time += 1
        time_d = self.time
        reward = 0
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        temp_torque = self.Compute_torque_action(action)
        print("temp_torque is:", temp_torque)
        now_torque = self.torque_state + temp_torque  # renormalize
        print("now torque_state is:", now_torque)

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
                return self.state, reward, True, False,{}
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
        print("joint_state_difs are:", joint_state_difs)
        dif_count = 0
        for dif in joint_state_difs:
            dif_count += 1
            if abs(dif) > 0.75 * PI and dif_count <= 6:  # the last joint is not important
                # print(self.joint_state_error)
                reward -= self.time if self.time<=10 else 10
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
                    reward += 10+self.count_checkpoint
                    self.count_done += 1
                else:
                    done = False
                    reward += 1
            else:
                if time_d <= self.ck_point_num:
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
        return self.state, reward, done, False,info



    def render(self):
        # 可选的渲染方法
        pass

    def _compute_reward(self, w_dist=0.4, w_joint=0.3, w_ori=0.2, w_vel=0.1):
        l_d, l_j, l_ori, l_v = 3.6, 0.8, 3, 3
        # l_d, l_j, l_ori, l_v = 15, 3, 4.5, 3.5 # initial hyperparameter
        # l_d, l_j, l_ori, l_v = 20, 5, 6, 5
        reward = 0

        if self.error_buf.at_brim():
            d_mean, j_mean, ori_mean, v_mean = self.error_buf.mean_error()
            l_d_new = kl_05 / d_mean
            l_j_new = kl_05 / j_mean
            l_ori_new = kl_05 / ori_mean
            l_v_new = kl_05 / v_mean
            l_d = l_d_new if l_d_new > l_d else l_d
            l_j = l_j_new if l_j_new > l_j else l_j
            l_ori = l_ori_new if l_ori_new > l_ori else l_ori
            l_v = l_v_new if l_v_new > l_v else l_v
            self.error_buf.flag_reset()

        # 根据末端执行器位置与目标位置之间的距离计算惩罚
        d_error = np.linalg.norm((self.state_xyz - self.target_xyz) / 1000)
        x_d = d_error
        d_loss = -1 + 2 / (
                np.exp(x_d * l_d) + np.exp(-1 * x_d * l_d))  # use a logistic smooth kernel function here, in [-1,0]
        reward += w_dist * d_loss
        print("dis reward:", d_loss)
        self.rms_error += (d_error * 1000) ** 2

        # 角度惩罚
        joint_error = np.linalg.norm(self.joint_state_error[0:5], ord=1)
        x_j = joint_error
        j_loss = -1 + 2 / (np.exp(x_j * l_j) + np.exp(-1 * x_j * l_j))
        print("joint reward:", j_loss)
        reward += w_joint * j_loss

        # 姿态惩罚
        delta_ori = self.quatdiff_in_euler(self.state_orientation, self.target_orientation)
        ori_error = np.linalg.norm(delta_ori, ord=1)
        x_ori = ori_error
        ori_loss = -1 + 2 / (np.exp(x_ori * l_ori) + np.exp(-1 * x_ori * l_ori))
        print("ori reward:", ori_loss)
        reward += w_ori * ori_loss

        # 速度惩罚, only when in the middle of the trajectory
        v_loss = 0
        if 2 < self.time < self.traj_length - 2:
            vel_error = np.linalg.norm(self.linear_vel - self.target_velocity, ord=1)
            x_v = vel_error
            v_loss = -1 + 2 / (np.exp(x_v * l_v) + np.exp(-1 * x_v * l_v))
            self.error_buf.store(d_error, joint_error, ori_error, vel_error)
        else:
            reward = reward / (w_dist + w_joint + w_ori)
        print("velocity reward:", v_loss)
        reward += w_vel * v_loss

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

    def trajectory(self, backtrace = 0):
        time_d = self.time
        if time_d <= self.ck_point_num:
            time = time_d - backtrace
        else:
            time = self.ck_point_num - backtrace
        return (self.samplePoint[time],self.sampleAngle[time]) # Attention here! This is taken in millimeter

    def Compute_torque_state(self,torque):
        torque_1234 = np.array([torque[0],torque[1],torque[2],torque[3]]) / 50
        torque_567 = np.array([torque[4],torque[5],torque[6]]) /10

        return np.hstack((torque_1234,torque_567))

    def Compute_torque_action(self,torque):
        torque_1234 = np.array([torque[0],torque[1],torque[2],torque[3]]) *5
        torque_567 = np.array([torque[4],torque[5],torque[6]])

        return np.hstack((torque_1234,torque_567))

    @staticmethod
    def check_xyz(xyz):
        if xyz[2] >= 0:
            return True
        else:
            return False