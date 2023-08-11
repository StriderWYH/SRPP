#!/usr/bin/python3

import sys
sys.path.append("/home/wyh/SRPP_franka/src/SRPP/scripts")
print(sys.path)
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
from gazebo_env_trajectory import *
from baselinepretraining import *
from baseline2 import *
from baseline3 import *
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems



pos_increment = 0.01


"""
Program run from here
"""
def main():

    # env = gazebo_env
    # env = FrankaEnv
    # env = trajectoryEnv
    # env = PretrainedEnv
    # env = PretrainedEnv2
    env = PretrainedEnv3
    hid = 256
    l = 2
    gamma = 0.99
    seed = 0
    epochs = 50
    exp_name = 'sac'

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    torch.set_num_threads(torch.get_num_threads())

    sac(env_fn=env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid] * l),
        steps_per_epoch=4000, epochs=30,gamma=0.99, seed=0,
        logger_kwargs=logger_kwargs, batch_size=100, start_steps=5000,
        update_after=1000, update_every=50, num_test_episodes=10,max_ep_len=200)



if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass












