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
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
pos_increment = 0.01


"""
Program run from here
"""
def main():
    # eval_env = PretrainedEnv3()
    env = PretrainedEnv3()

    # Check the validation of the customized env
    check_env(env, warn=True, skip_render_check=True)
    policy_kwargs = dict(net_arch=[256, 256])
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=4000,
        save_path="./src/SRPP/scripts/logs",
        name_prefix="straightLine",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(env, best_model_save_path="./src/SRPP/scripts/logs/sac_Franka_best_model",
                             log_path="./src/SRPP/scripts/logs/sac_Franka_eval", eval_freq=4000,n_eval_episodes = 5,
                             deterministic=True, render=False)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create network model or load a pretained model
    model = SAC("MlpPolicy", env, learning_starts = 5000, batch_size=100,learning_rate= 1e-3,ent_coef='auto_0.2',policy_kwargs =policy_kwargs,
                train_freq = (50,"step"),gradient_steps=1,target_update_interval=1,verbose=1, tensorboard_log="./src/SRPP/scripts/logs/sac_Franka_data/",
                _init_setup_model=True)
    # model = SAC.load("sac_Franka_model")

    # Train the network
    model.learn(total_timesteps=200000, log_interval=10, tb_log_name="StraightLine_Tracking", callback=callback)
    # Save the network at the end
    model.save("./src/SRPP/scripts/logs/sac_Franka_model")

    # An infinite Test
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated,truncated,info = env.step(action)
        if terminated:
            obs= env.reset()



if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass












