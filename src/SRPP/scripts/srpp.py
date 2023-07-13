#!/usr/bin/env python3

import sys
import copy
import time
import rospy
import gym
from gym import spaces
import numpy as np
from project_header import *
from sac import sac
import core as core
from gazebo_env import *
from gazebo_env_torque import *
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems



pos_increment = 0.01

def test():
    # Initialize ROS node
    rospy.init_node('sacnode')

    limb = PandaArm()

    while (rospy.is_shutdown()):
        print("ROS is shutdown!")

    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 恢复仿真
    pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停仿真
    reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

    # reset the simulation world
    rospy.wait_for_service("/gazebo/reset_world")
    try:
        reset_proxy()
    except rospy.ServiceException as e:
        print("/gazebo/reset_simulation service call failed")

    neutral_joint_state = np.array([limb._neutral_pose_joints[n] for n in limb._joint_names])  # the whole 7 joints
    limb.exec_position_cmd(neutral_joint_state)
    time.sleep(2)
    joint_state = np.array([limb._joint_angle[n] for n in limb._joint_names])
    print("neutral position:", joint_state, "\n")
    print("neutral joint efforts:",limb._joint_effort, "\n")
    linear_vel = limb.endpoint_velocity()['linear']
    print("neutral endpoint_velocity", linear_vel, "\n")
    #
    rospy.wait_for_service("/gazebo/pause_physics")
    try:
        pass
        pause()
    except (rospy.ServiceException) as e:
        print("/gazebo/pause_physics service call failed")

    # before unpause, publish the news
    # change the efforts
    # action_space = spaces.Box(low=-7.0, high=7.0, shape=(7,), dtype=np.float32)
    # a = action_space.sample()
    # a += limb.gravity_comp() + limb.coriolis_comp()
    # a = np.array([-10, -5, -5, -5, -15, -5, -5])
    a = np.array([-10, 0, 0, 0, 0, 0, 0])
    print("effort value a is", a, "\n")
    limb.exec_torque_cmd(a)
    time.sleep(0.1)
    joint_before_state = np.array([limb._joint_angle[n] for n in limb._joint_names])
    print("before unpause:", joint_before_state)
    print("before unpause joint efforts:", limb._joint_effort,"\n")
    linear_vel = limb.endpoint_velocity()['linear']
    print("before unpause endpoint_velocity", linear_vel, "\n")
    rospy.wait_for_service("/gazebo/unpause_physics")
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print("/gazebo/unpause_physics service call failed")

    #
    time.sleep(0.1)
    #
    rospy.wait_for_service("/gazebo/pause_physics")
    try:
        pass
        pause()
    except (rospy.ServiceException) as e:
        print("/gazebo/pause_physics service call failed")

    joint_after_state = np.array([limb._joint_angle[n] for n in limb._joint_names])
    print("after unpause:", joint_after_state)
    print("after unpause joint efforts:", limb._joint_effort, "\n")
    linear_vel = limb.endpoint_velocity()['linear']
    print("after unpause endpoint_velocity", linear_vel, "\n")



"""
Program run from here
"""
def main():

    # test()


    env = FrankaEnv
    hid = 256
    l = 2
    gamma = 0.99
    seed = 0
    epochs = 20
    exp_name = 'sac'

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    torch.set_num_threads(torch.get_num_threads())

    sac(env_fn=env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid] * l),
            gamma=gamma, seed=seed, epochs=epochs,
            logger_kwargs=logger_kwargs)



if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass












