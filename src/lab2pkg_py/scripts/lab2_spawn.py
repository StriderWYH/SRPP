#!/usr/bin/env python

import rospy
import rospkg
import os
import sys
import yaml
import random
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

yamlpath = 'lab2_data.yaml'

if __name__ == '__main__':

    # Initialize rospack
    rospack = rospkg.RosPack()
    # Get path to yaml
    lab2_path = rospack.get_path('lab2pkg_py')
    yamlpath = os.path.join(lab2_path, 'scripts', 'lab2_data.yaml')

    with open(yamlpath, 'r') as f:
        try:
            # Load the data as a dict
            data = yaml.load(f)
            # Load block position
            block_xy_pos = data['block_xy_pos']
            
        except:
            sys.exit()

    # Initialize ROS node
    rospy.init_node('ur3_gazebo_spawner', anonymous=True)
    # Initialize ROS pack
    rospack = rospkg.RosPack()
    # Get path to block
    ur_path = rospack.get_path('ur_description')
    block_path = os.path.join(ur_path, 'urdf', 'block.urdf')
    block1_path = os.path.join(ur_path, 'urdf', 'block_green.urdf')
    block2_path = os.path.join(ur_path, 'urdf', 'block_green.urdf')
    block3_path = os.path.join(ur_path, 'urdf', 'block_green.urdf')
    block_paths = [block1_path, block2_path, block3_path]
    # Wait for service to start
    rospy.wait_for_service('gazebo/spawn_urdf_model')
    spawn = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
    delete = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)

    x_ranlist = [0,0,0,0,0]
    y_ranlist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(5):
        x_ranlist[i] = 0.04 + (i+1) * 0.035
    for j in range(20):
        y_ranlist[j] = -0.3 + (j+1) * 0.035
    
    random.shuffle(x_ranlist)
    random.shuffle(y_ranlist)
        
    choice_user = raw_input("Spawn randomly or not < 1 for random>: ")
    # if int(choice_user) == 1:
    #     xy_ran = []

    #     num_block_deleted = int(raw_input("Enter the desired deleted blocks number: "))
    #     for j in range(num_block_deleted):
    #         block_name = 'block' + str(j + 1)
    #         delete(block_name)

        
    #     num_block_spawned = int(raw_input("Enter the desired spawned blocks number: "))
    #     if num_block_spawned < 0 | num_block_spawned > 10:
    #         num_block_spawned = 10
    #     for i in range(num_block_spawned):
    #         block_name = 'block' + str(i+1)

    #         x_ran = x_ranlist[i]
    #         y_ran = y_ranlist[i]
    #         xy_ran.append((x_ran,y_ran))
    #         pose = Pose(Point(x_ran, y_ran, 0), Quaternion(0, 0, 0, 0))
    #         spawn(block_name, open(block_paths[random.randint(0,2)], 'r').read(), 'block', pose, 'world')
    # else:
    # Starting location ?
    starting_location = None
    while not starting_location:
        starting_location = raw_input("Enter starting location number <Either 1 2 or 3>: ")
        starting_location = int(starting_location)
        if (starting_location != 1) and (starting_location != 2) and (starting_location != 3):
            starting_location = None
            print("Wrong input \n\n")

        # 0-indexed
        starting_location -= 1

        # Missing block ?
        missing_block = None
        while missing_block is None:
            missing_block = raw_input("Missing Block?(y/n): ")
            missing_block = str(missing_block)
            if (missing_block != 'y') and (missing_block != 'n'):
                missing_block = None
                print("Wrong input \n\n")
            
        missing_block = (missing_block == 'y')

        # Delete previous blocks
        for height in range(3):
            block_name = 'block' + str(height + 1)
            delete(block_name)


        if not missing_block:
            # Spawn three blocks
            for height in range(3):
                block_name = 'block' + str(height + 1)
                pose = Pose(Point(block_xy_pos[starting_location][height][0], 
                                block_xy_pos[starting_location][height][1], 0), Quaternion(0, 0, 0, 0))
                spawn(block_name, open(block_paths[2-height], 'r').read(), 'block', pose, 'world')
        
        else:
            missing_block_height = random.randint(0, 2)
            # Spawn two blocks
            for height in range(3):
                if height == missing_block_height:
                    continue
                block_name = 'block' + str(height + 1)
                pose = Pose(Point(block_xy_pos[starting_location][height][0], 
                                block_xy_pos[starting_location][height][1], 0), Quaternion(0, 0, 0, 0))
                spawn(block_name, open(block_paths[2-height], 'r').read(), 'block', pose, 'world')

