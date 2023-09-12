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
# from Joint_circle_trajectory import *
from Torque_Circle_ref import *
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback,BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
"""
Program run from here
"""
class RMSCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0,env=None):
        super(RMSCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.env.rms_record > 0:
            self.logger.record('Error/RMS d_error', self.env.error[0])
            self.logger.record('Error/j_error_mean', self.env.error[1])
            self.logger.record('Error/o_error_mean', self.env.error[2])
            self.logger.record('Error/v_error_mean', self.env.error[3])
            self.logger.record('Error/max_dist_error', self.env.error[4])

        return True


def main():
    steps_every_epoch = 4000
    epochs = 60
    total_steps = steps_every_epoch * epochs
    k_update_times = 5
    update_k_everysteps = int(total_steps / k_update_times)
    save_frequency = steps_every_epoch*2
    eval_frequency = steps_every_epoch/2

    env = TorqueCircleTrajectory(update_k_everysteps=update_k_everysteps)
    # test_env = CircleTrajectoryTest(update_k_everysteps=update_k_everysteps)
    # Check the validation of the customized env
    check_env(env, warn=True, skip_render_check=True)

    # model_single = SAC.load("./src/SRPP/scripts/logs/Torque_Tracking_R1_48000_steps_single", env=env)
    # model_single = SAC.load("./src/SRPP/scripts/logs/Torque_Tracking_R1_part1_32000_steps", env=env)
    # writer_single = SummaryWriter('./src/SRPP/scripts/logs/sac_test_result/Part1')
    #
    # obs, info = env.reset()
    # episode = 0
    # test_episode = 20
    # while episode < test_episode:
    #     action, _states = model_single.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         episode += 1
    #     if env.rms_record > 0:
    #         writer_single.add_scalar('Error/RMS d_error', env.error[0], episode)
    #         writer_single.add_scalar('Error/j_error_mean', env.error[1], episode)
    #         writer_single.add_scalar('Error/o_error_mean', env.error[2], episode)
    #         writer_single.add_scalar('Error/v_error_mean', env.error[3], episode)
    #         writer_single.add_scalar('Error/max_dist_error', env.error[4], episode)
    #
    # model_single = SAC.load("./src/SRPP/scripts/logs/Torque_Tracking_R1_part2_32000_steps", env=env)
    # writer_single = SummaryWriter('./src/SRPP/scripts/logs/sac_test_result/Part2')
    #
    # obs, info = env.reset()
    # episode = 0
    # test_episode = 20
    # while episode < test_episode:
    #     action, _states = model_single.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, info = env.reset()
    #         episode += 1
    #     if env.rms_record > 0:
    #         writer_single.add_scalar('Error/RMS d_error', env.error[0], episode)
    #         writer_single.add_scalar('Error/j_error_mean', env.error[1], episode)
    #         writer_single.add_scalar('Error/o_error_mean', env.error[2], episode)
    #         writer_single.add_scalar('Error/v_error_mean', env.error[3], episode)
    #         writer_single.add_scalar('Error/max_dist_error', env.error[4], episode)


    model_1 = SAC.load("./src/SRPP/scripts/logs/Torque_Tracking_R1_part1_32000_steps", env=env)
    model_2 = SAC.load("./src/SRPP/scripts/logs/Torque_Tracking_R1_part2_32000_steps", env=env)
    writer = SummaryWriter('./src/SRPP/scripts/logs/sac_test_result/Combined')

    obs, info = env.reset()
    t = 0
    episode = 0
    test_episode = 20
    while episode < test_episode:
        action_1, _states_2 = model_1.predict(obs, deterministic=True)
        action_2, _states_2 = model_2.predict(obs, deterministic=True)
        if t < 80:
            obs, reward, terminated, truncated, info = env.step(action_1)
            t += 1
        else:
            obs, reward, terminated, truncated, info = env.step(action_2)
            t += 1
        if terminated or truncated:
            obs, info = env.reset()
            t = 0
            episode += 1
        if env.rms_record > 0:
            writer.add_scalar('Error/RMS d_error', env.error[0], episode)
            writer.add_scalar('Error/j_error_mean', env.error[1], episode)
            writer.add_scalar('Error/o_error_mean', env.error[2], episode)
            writer.add_scalar('Error/v_error_mean', env.error[3], episode)
            writer.add_scalar('Error/max_dist_error', env.error[4], episode)


if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass
