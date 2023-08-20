#!/usr/bin/python3
import sys

sys.path.append("/home/wyh/SRPP_franka/src/SRPP/scripts")
sys.path.append("/home/wyh/SRPP_franka/src/SRPP/scripts/stable_baselines3")
print(sys.path)
import copy
import time
import rospy
import numpy as np
from baseline3 import *
from baseline4 import *
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

"""
Program run from here
"""


def main():
    steps_every_epoch = 4000
    epochs = 50
    total_steps = steps_every_epoch * epochs
    k_update_times = 10
    update_k_everysteps = int(total_steps / k_update_times)
    save_frequency = update_k_everysteps
    eval_freqency = steps_every_epoch

    env = Line4(update_k_everysteps=update_k_everysteps)
    # Check the validation of the customized env
    check_env(env, warn=True, skip_render_check=True)

    model = SAC.load("./src/SRPP/scripts/logs/sac_Franka_model" + f"_{11}", env=env)
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        render=False,
        deterministic=True,
        return_episode_rewards=True,
        warn=True,
        callback=None
    )
    print("episode_rewards:", episode_rewards)
    print("episode_lengths", episode_lengths)


if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass
