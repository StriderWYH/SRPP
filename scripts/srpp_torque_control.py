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
    # check_env(test_env, warn=True, skip_render_check=True)
    policy_kwargs = dict(net_arch=[256, 256])
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_frequency,
        save_path="./src/SRPP/scripts/logs",
        name_prefix="Torque_Tracking_R1_part2",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(env, best_model_save_path="./src/SRPP/scripts/logs/sac_Franka_best_model_Torque_Tracking_R1",
                                 log_path="./src/SRPP/scripts/logs/sac_Franka_eval", eval_freq=eval_frequency,
                                 n_eval_episodes=5, deterministic=True, render=False)
    rms_callback = RMSCallback(env=env)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, rms_callback])

    # Initialize the kernel parameter
    o, _ = env.reset()
    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    ep_rms_error = 0
    while n < 20:
        d = env.pre_train_step()
        # print("round:",n, "\n")
        if d:
            o, _ = env.reset()
            n += 1
    env.error_buf.flag_set()
    # Create network model or load a pretrained model
    model = SAC("MlpPolicy", env, learning_starts=4000, batch_size=160, learning_rate=1e-3, ent_coef='auto_0.2',
                policy_kwargs=policy_kwargs, tau=0.005, gamma=0.95, train_freq=(30, "step"), gradient_steps=-1,
                buffer_size=200000,
                target_update_interval=1, verbose=1, tensorboard_log="./src/SRPP/scripts/logs/sac_Franka_data/R1_240P_part2",
                action_noise=None, use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, _init_setup_model=True)

    # model = SAC.load("./src/SRPP/scripts/logs/sac_Franka_modelCircle_7", env=env)
    # model.load_replay_buffer("./src/SRPP/scripts/logs/sac_replay_buffer")
    model.learn(total_timesteps=total_steps, log_interval=3, tb_log_name="Torque_Tracking_R1_part2",
                callback=callback,
                reset_num_timesteps=True)
    model.save("./src/SRPP/scripts/logs/sac_Franka_model" + "Torque_Tracking_R1")
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
