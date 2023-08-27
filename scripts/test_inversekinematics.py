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

PI = np.pi
time_delta = 0.008


class PD:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.pre_pose_error = 0
        self.last_time = time.time()
    def control(self, pose_error):
        # current_time = time.time()
        current_time = time.time()
        # delta_time = current_time - self.last_time
        # print("delta time is :", delta_time, "\n")
        delta_time = time_delta
        derivative = (pose_error - self.pre_pose_error)

        output = self.kp * pose_error + self.kd * derivative

        self.pre_pose_error = pose_error
        self.last_time = current_time
        return output

    def reset(self):
        self.pre_pose_error = 0


def main():
    # Initialize ROS node
    rospy.init_node('sacnode')
    rms_error_x = 0
    pd_ctrl = PD(1, 0.01)
    limb = PandaArm()
    while (rospy.is_shutdown()):
        print("ROS is shutdown!")
    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 恢复仿真
    pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停仿真

    joint_state = np.array(
        [limb._neutral_pose_joints[n] for n in limb._joint_names])  # the whole 7 joints
    print("joint state is :", joint_state)
    limb.exec_position_cmd(joint_state)  # move to the neutral position
    time.sleep(2.0)
    target_orientation = limb.endpoint_pose()['orientation']  # this is am array of type quaternion
    print("target orientation is :", target_orientation, "\n")
    xyz = np.array([700, 0, 400])
    temp_check = limb.inverse_kinematics(xyz / 1000,
                                         target_orientation)
    limb.exec_position_cmd(temp_check[1])
    time.sleep(1)
    samplePoint = []  # millimeter
    sampleAngle = []
    length = 1249
    for t in range(0, length + 1):
        r = 300
        x = r * np.cos(0.0016 * PI * t) + r + 100
        y = r * np.sin(0.0016 * PI * t)
        samplePoint.append(np.array([x, y, 400]))
        xyz = np.array([x, y, 400])
        temp_check = limb.inverse_kinematics(samplePoint[t] / 1000,
                                             target_orientation)
        assert temp_check[0], "position t has no inverse kinematics"
        joint_state = np.array([limb._joint_angle[n] for n in limb._joint_names])
        joint_error = temp_check[1] - joint_state

        sampleAngle.append(temp_check[1])
        print("target angle is:", temp_check[1])
        ref_angle_vel = pd_ctrl.control(joint_error)/time_delta
        tem = joint_error / time_delta
        print("angle_vel is:", tem)
        print("ref_angle_vel is:", ref_angle_vel)

        # limb.exec_velocity_cmd(ref_angle_vel)
        limb.exec_position_cmd(temp_check[1])
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        # # 接下来需要停一小段时间，让小车以目前的速度行驶一段时间，TIME_DELTA = 0.1s
        time.sleep(time_delta)
        # rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        state_xyz = 1000 * limb.endpoint_pose()['position']
        xyz_error = state_xyz - xyz
        print("target xyz is:", xyz)
        print("current xyz is:", state_xyz)
        print("xyz_error is:", xyz_error, "\n")
        rms_error_x += np.abs(xyz_error[0])**2
        # # 首先unpause，发布消息后，开始仿真

    rms_error = np.sqrt(rms_error_x/length)
    print("rms error x:", rms_error)

if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass
