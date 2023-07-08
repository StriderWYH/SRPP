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
import torch
from torch.optim import Adam

from panda_robot import PandaArm
from franka_dataflow.getch import getch
from future.utils import viewitems



pos_increment = 0.01
# ori_increment = 0.001
#
# def map_keyboard():
#     """
#         Map keyboard keys to robot joint motion. Keybindings can be
#         found when running the script.
#     """
#
#     limb = PandaArm()
#
#     has_gripper = limb.get_gripper() is not None
#
#     def set_ee_target(action, value):
#         pos, ori = limb.ee_pose()
#
#         if action == 'position':
#             pos += value
#         status, j_des = limb.inverse_kinematics(pos, ori)
#         if status:
#             limb.exec_position_cmd(j_des)
#
#     def set_g(action):
#         if has_gripper:
#             if action == "close":
#                 limb.get_gripper().close()
#             elif action == "open":
#                 limb.get_gripper().open()
#             elif action == "calibrate":
#                 limb.get_gripper().calibrate()
#     def reset_robot(args):
#         limb.untuck()
#
#     bindings = {
#         '5': (set_ee_target, ['position', np.asarray([pos_increment, 0, 0])], "x increase"),
#         '2': (set_ee_target, ['position', np.asarray([-pos_increment, 0, 0])], "x decrease"),
#         '1': (set_ee_target, ['position', np.asarray([0, pos_increment, 0])], "y increase"),
#         '3': (set_ee_target, ['position', np.asarray([0, -pos_increment, 0])], "y decrease"),
#         '7': (set_ee_target, ['position', np.asarray([0, 0, pos_increment])], "z increase"),
#         '4': (set_ee_target, ['position', np.asarray([0, 0, -pos_increment])], "z decrease"),
#         'r': (reset_robot, [None], "reset to neutral pose")
#      }
#     if has_gripper:
#         bindings.update({
#         '8': (set_g, "close", "close gripper"),
#         '9': (set_g, "open", "open gripper"),
#         'i': (set_g, "calibrate", "calibrate gripper")
#         })
#     done = False
#     rospy.logwarn("Controlling end-effector position. Press ? for help, Esc to quit. For ease, use the numeral keys in the numberpad of the keyboard.\n\nWARNING: The motion will be slightly jerky!!\n")
#     while not done and not rospy.is_shutdown():
#         c = getch()
#         if c:
#             #catch Esc or ctrl-c
#             if c in ['\x1b', '\x03']:
#                 done = True
#                 rospy.signal_shutdown("Example finished.")
#             elif c in bindings:
#                 cmd = bindings[c]
#                 if c == '8' or c == 'i' or c == '9':
#                     cmd[0](cmd[1])
#                     print("command: %s" % (cmd[2],))
#                 else:
#                     #expand binding to something like "set_j(right, 'j0', 0.1)"
#                     cmd[0](*cmd[1])
#                     print("command: %s" % (cmd[2],))
#             else:
#                 print("key bindings: ")
#                 print("  Esc: Quit")
#                 print("  ?: Help")
#                 for key, val in sorted(viewitems(bindings),
#                                        key=lambda x: x[1][2]):
#                     print("  %s: %s" % (key, val[2]))


"""
Program run from here
"""
def main():

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

    #
    # print("Initializing node... ")
    # rospy.init_node("fri_example_joint_position_keyboard")
    # print("Getting robot state... ")
    #
    # def clean_shutdown():
    #     print("\nExiting example.")
    #
    # rospy.on_shutdown(clean_shutdown)
    #
    # map_keyboard()
    # print("Done.")

if __name__ == '__main__':

    try:
        main()
    # When Ctrl+C is executed, it catches the exception
    except rospy.ROSInterruptException:
        pass












