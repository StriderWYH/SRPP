#!/usr/bin/python3
import sys
import copy
import time
import rospy
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
# from gym.envs.classic_control import rendering
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

target_obj = np.array([300,50,150]) # millimeter

class trajectoryEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    def __init__(self, thr=10, weight=0.00069,vel=4.0,accel=4.0):

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)				#恢复仿真
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)				    #暂停仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)	
        self.target_xyz = 0
        self.time = 0
        # set the initial start pos
        
    

        # 定义观测空间
        self.freeze_step = 0
        self.torque_space = spaces.Box(low = obs_torque_low_bd, high=obs_torque_high_bd, dtype=np.float64)

        self.observation_space = spaces.Box(low = -10.0, high= 10.0, shape=(27,), dtype=np.float64)
        # the observation space we define contains the following things:
        # end effector position; end effector orientation; end effector linear velocity; end effector angular velocity; joint state
        # 3 + 4 + 3 + 3 + 7 = 20
        # 定义动作空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # temporary action torque
        
        # Initialize ROS node
        rospy.init_node('sacnode')
        
        self.limb = PandaArm()

        while (rospy.is_shutdown()):
            print("ROS is shutdown!")
        
        self.joint_state = np.array([self.limb._neutral_pose_joints[n] for n in self.limb._joint_names]) # the whole 7 joints
        self.limb.exec_position_cmd(self.joint_state)
        time.sleep(2.0)
        self.target_orientation = self.limb.endpoint_pose()['orientation']
        xyz = self.trajectory(backtrace=1)/1000 #  Attention here , this is taken in meter
        inverse = self.limb.inverse_kinematics(xyz,self.limb.endpoint_pose()['orientation'])
        assert inverse[0] == True, "Flase inverse kinematics"
        self.init_joint_state = inverse[1]
        # 初始化环境
        self.reset()

        # print(self.state_orientation)
        self.thr = thr                 # hyperparameter, in millimeter
        self.weight = weight              # hyperparameter
        self.current_error = -math.inf
        self.count = 0
        self.count_done = 0
        self.seed()

        self.vel = vel
        self.accel = accel

    def reset(self, startstate=None):
        self.count_checkpoint = 0
        self.rms_error = 0
        self.freeze_step = 0
        self.freeze_pos_step = 0
        self.time = 0 # 这个是记录机械臂时间t的

        # 初始化机械臂状态和目标位置
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        self.limb.enable_robot()
        time.sleep(2.0)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        self.move_to_start()
        self.torque_state = self.limb.gravity_comp() + self.limb.coriolis_comp()
        self.state_xyz = 1000*self.limb.endpoint_pose()['position']
        self.last_xyz = self.state_xyz
        self.target_xyz = self.trajectory()
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is a array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        self.angular_vel = self.limb.endpoint_velocity()['angular']

        # temp = [self.state_xyz[0]/100, self.state_xyz[1]/100, self.state_xyz[2]/100, self.state_orientation.x, self.state_orientation.y,
        #               self.state_orientation.z, self.state_orientation.w, self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
        #               self.angular_vel[0], self.angular_vel[1], self.angular_vel[2]]
        # self.state = np.hstack((self.joint_state, self.torque_state / 10))
        """
                    Baseline added
        """
        temp = [self.state_xyz[0] / 100, self.state_xyz[1] / 100, self.state_xyz[2] / 100,
                self.target_xyz[0] / 100, self.target_xyz[1] / 100, self.target_xyz[2] / 100,
                self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
                self.state_orientation.x, self.state_orientation.y,
                self.state_orientation.z, self.state_orientation.w
                ]
        state = np.hstack((self.joint_state, self.torque_state / 10))
        self.state = np.hstack((state, temp))
        # 返回初始状态
        # print(self.state)
        # print("\n")
        return self.state


    """
        This function is used to move arm to the start of the trajectory, the joint state will be automatically updated
    """
    def move_to_start(self):
        self.joint_state = self.init_joint_state
        self.limb.exec_position_cmd(self.joint_state) # move to the start of the trajectory
        time.sleep(2.0)
        return

    
    
    
    def update_robot_state(self):
        
        self.last_xyz = self.state_xyz
        self.joint_state = np.array([self.limb._joint_angle[n] for n in self.limb._joint_names]) # the whole 7 joints
        self.state_xyz = 1000*self.limb.endpoint_pose()['position']
        self.state_orientation = self.limb.endpoint_pose()['orientation']  # this is a array of type quaternion
        self.linear_vel = self.limb.endpoint_velocity()['linear']
        self.angular_vel = self.limb.endpoint_velocity()['angular']
        self.target_xyz = self.trajectory()
        # temp = [self.state_xyz[0], self.state_xyz[1], self.state_xyz[2], self.state_orientation.x, self.state_orientation.y,
        #               self.state_orientation.z, self.state_orientation.w, self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
        #               self.angular_vel[0], self.angular_vel[1], self.angular_vel[2]]
        # self.state = np.hstack((self.joint_state, self.torque_state / 10))
        """
            Baseline added
        """
        temp = [self.state_xyz[0] / 100, self.state_xyz[1] / 100, self.state_xyz[2] / 100,
                self.target_xyz[0] / 100, self.target_xyz[1] / 100, self.target_xyz[2] / 100,
                self.linear_vel[0], self.linear_vel[1], self.linear_vel[2],
                self.state_orientation.x, self.state_orientation.y,
                self.state_orientation.z, self.state_orientation.w
                ]
        state = np.hstack((self.joint_state, self.torque_state / 10))
        self.state = np.hstack((state, temp))


    

    def step(self, action, move = True):
        # 更新机械臂状态
        self.time += 1
        reward = 0
        distance_reward, velocity_reward, ori_reward, torque_reward = 0, 0, 0, 0
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        now_torque = self.torque_state + action*7  # renormalize to -7->+7

        if self.torque_space.contains(now_torque):
            self.freeze_step = 0
            self.torque_state = now_torque
            # print("exec torque")
            self.limb.exec_torque_cmd(now_torque)
        else: 
            self.freeze_step += 1
            if self.freeze_step >= 10:
                reward -= 400
                # info = {
                #     'target_position': self.target_xyz,
                #     'current_position': self.state_xyz,
                #     'current_obs': self.state,
                #     'distant_ret': distance_reward,
                #     'velocity_ret': velocity_reward,
                #     'orientation_ret': ori_reward,
                #     'torque_ret': torque_reward
                # }
                # return self.state, reward, True, info
            self.limb.exec_torque_cmd(self.torque_state)
            reward -= 40

        #### this one is the penalty for the not move action, I temporary loss it 

        # if np.linalg.norm(self.state_xyz - self.last_xyz) < 30:
        #     self.freeze_pos_step += 1
        #     if self.freeze_pos_step >= 20:
        #         reward -= 400
        # else:
        #     self.freeze_pos_step = 0


        # 首先unpause，发布消息后，开始仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # 接下来需要停一小段时间，让机械臂以目前的力矩执行一段时间，TIME_DELTA = 0.1s
        time.sleep(time_delta)
        # 然后我们停止仿真，开始观测目前的状态
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.update_robot_state()

        # check the reward term
        if self.limb.in_safe_state() and self.check_xyz(self.state_xyz):  # check whether the robot is fine
            distance = np.linalg.norm(self.state_xyz - self.target_xyz)  # Euclidean distance
            pre_targxyz = self.trajectory(backtrace=1)
            if distance <= self.thr:
                if self.time <= 65: # additional 5 step is given for chance
                    self.count_checkpoint += 1
                    if self.time >= 60:
                        done = True
                        reward += 100
                        reward += self.count_checkpoint * ( 1 + self.count_checkpoint/60) # if the final trajectory is finished, we give the final extra reward
                        self.count_done += 1
                        reward -= self.rms_print()  # finally we count the rms value of the whole real trajectory, then use it as the loss penalty
                    else:
                        done = False
                        if np.linalg.norm(self.last_xyz - pre_targxyz) <= self.thr:
                            reward += 30
                        reward += 100
                else: # not reaching the final destination
                    done = True
                    self.rms_print()
                    #reward -= self.rms_print()
                
            else:
                if self.time <= 65:
                    done = False
                else:
                    done = True
                    self.rms_print()
                    #reward -= self.rms_print()

            # 计算奖励
            reward_delta, distance_reward, velocity_reward, ori_reward, torque_reward = self._compute_reward()
            reward += reward_delta
        else:
            print("not safe state\n")
            done = True
            reward -= -400
             # reinitialize the world

        info = {
            'target_position': self.target_xyz,
            'current_position': self.state_xyz,
            'current_obs': self.state,
            'distant_ret': distance_reward,
            'velocity_ret': velocity_reward,
            'orientation_ret': ori_reward,
            'torque_ret': torque_reward
        }


        # 返回下一个状态、奖励、是否终止以及其他信息
        return self.state, reward, done, info

    def render(self):
        # 可选的渲染方法
        pass

    def rms_print(self):
        print("\n\n\n")
        rms_result =  np.sqrt(self.rms_error/self.time)
        print("The rms error during this round is:  ",rms_result)
        print("\nThe number of checkpoints reached this round is:  ", self.count_checkpoint)
        print("\n\n\n")
        return rms_result

    def _compute_reward(self):
        # 根据末端执行器位置与目标位置之间的距离计算奖励
        distance = np.linalg.norm(self.state_xyz - self.target_xyz)/50
        self.rms_error += (distance*50) ** 2
        distance_reward = -distance  # 使用负距离作为奖励的一部分，目标是最小化距离

        # 加入关节速度奖励部分
        # velocity_reward = -np.sum(self.linear_vel ** 2)  # 使用关节速度的负平方作为奖励的一部分
        velocity_reward = - 0.01* np.sum(self.linear_vel ** 2)

        # we need to give some constraint to the orientation
        delta_ori = self.quatdiff_in_euler(self.state_orientation, self.target_orientation)
        ori_loss = -100*np.linalg.norm(delta_ori)  # every time 8
        ori_reward = ori_loss

        # we should try best to not let the torque too big
        torque_loss = -np.linalg.norm(self.torque_state)*0.01
        torque_reward = torque_loss

        reward = distance_reward + velocity_reward + ori_reward + torque_reward
        return reward, distance_reward, velocity_reward, ori_reward, torque_reward


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
        if self.time <= 60:
            time = self.time - backtrace
        else:
            time = 60 - backtrace
        return np.array([500, -290 + 10*time, 400]) # Attention here! This is taken in millimeter


    @staticmethod
    def check_xyz(xyz):
        if xyz[2]>=0:
            return True
        else:
            return False


    
