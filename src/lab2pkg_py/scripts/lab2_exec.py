#!/home/ur3/anaconda3/envs/spinningup/bin/python3

import sys
import copy
import time
import rospy
import gym
from gym import spaces
import numpy as np
from project_header import *
from project_func import *
#from blob_search import *
from sac import sac
import core as core
from gazebo_env import *
import torch
from torch.optim import Adam


"""
Program run from here
"""
def main():
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--exp_name', type=str, default='sac')
    # args = parser.parse_args()

    env = gazebo_env
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












