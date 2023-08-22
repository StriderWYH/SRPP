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
from CircleTrajectory import *
from TrajectoryTestEnv import *
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
    epochs = 200
    total_steps = steps_every_epoch * epochs
    k_update_times = 10
    update_k_everysteps = int(total_steps / k_update_times)
    save_frequency = update_k_everysteps
    eval_frequency = steps_every_epoch

    env = CircleTrajectory(update_k_everysteps=update_k_everysteps)
    test_env = CircleTrajectoryTest(update_k_everysteps=update_k_everysteps)
    # Check the validation of the customized env
    check_env(env, warn=True, skip_render_check=True)
    check_env(test_env, warn=True, skip_render_check=True)
    policy_kwargs = dict(net_arch=[256, 256])
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_frequency,
        save_path="./src/SRPP/scripts/logs",
        name_prefix="CircleTest",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(test_env, best_model_save_path="./src/SRPP/scripts/logs/sac_Franka_best_model_circletest",
                                 log_path="./src/SRPP/scripts/logs/sac_Franka_eval", eval_freq=eval_frequency,
                                 n_eval_episodes=5, deterministic=True, render=False)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create network model or load a pretrained model
    model = SAC("MlpPolicy", env, learning_starts=4000, batch_size=160, learning_rate=1e-3, ent_coef='auto_0.2',
                policy_kwargs=policy_kwargs, tau=0.005, gamma=0.99, train_freq=(30, "step"), gradient_steps=-1,
                buffer_size=200000,
                target_update_interval=1, verbose=1, tensorboard_log="./src/SRPP/scripts/logs/sac_Franka_data/",
                action_noise=None, use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, _init_setup_model=True)

    # model = SAC.load("./src/SRPP/scripts/logs/sac_Franka_modelCircle_7", env=env)
    # model.load_replay_buffer("./src/SRPP/scripts/logs/sac_replay_buffer")
    model.learn(total_timesteps=update_k_everysteps, log_interval=10, tb_log_name="Circle_Tracking_Test",
                callback=callback,
                reset_num_timesteps=True)
    model.save("./src/SRPP/scripts/logs/sac_Franka_model" + "CircleTest_1")
    # model.save_replay_buffer("./src/SRPP/scripts/logs/sac_replay_buffer")

    for i in range(2, k_update_times + 1):
        # update the kernel parameter
        env.error_buf.flag_set()
        # Train the network
        model = SAC.load("./src/SRPP/scripts/logs/sac_Franka_model" + f"CircleTest_{i-1}", env=env)
        model.learn(total_timesteps=update_k_everysteps, log_interval=10, tb_log_name="Circle_Tracking_Test",
                    callback=callback,
                    reset_num_timesteps=False)

        # Save the network at the end
        model.save("./src/SRPP/scripts/logs/sac_Franka_model" + f"CircleTest_{i}")
        # model.save_replay_buffer("./src/SRPP/scripts/logs/sac_replay_buffer")

    # A simple Test
    # model = SAC.load("./src/SRPP/scripts/logs/sac_Franka_best_model/best_model",env=env)
    # episode_rewards, episode_lengths = evaluate_policy(
    #     model,
    #     env,
    #     n_eval_episodes=10,
    #     render=False,
    #     deterministic=True,
    #     return_episode_rewards=True,
    #     warn=True,
    #     callback=None
    # )
    # print("episode_rewards:", episode_rewards)
    # print("episode_lengths", episode_lengths)


if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass
