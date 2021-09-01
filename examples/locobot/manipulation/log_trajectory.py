# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example for commanding robot with the src without using planner
"""
import datetime
import csv
import os
import argparse
from pyrobot import Robot

def main():

    parser = argparse.ArgumentParser(description='log trajectory csv file to control robot joint state')

    parser.add_argument('path', type=str, help="input trajectory csv file")
    parser.add_argument('--fps', type=int, default=10 , help="log csv fps")
    parser.add_argument('--ofps', type=int, default=1, help="output fps")
    args = parser.parse_args()

    target_joints = []

    with open(args.path) as csvfile:

        rows = csv.reader(csvfile)

        for row in rows:
            target_joints.append(row)

    bot = Robot('locobot')
    bot.arm.go_home()
    
    for joint in target_joints[1:-1:args.fps/args.ofps]:

        bot.arm.set_joint_positions([float(x) for x in joint[0:5]], plan=False)
        gripper_value = float(joint[5])
        if gripper_value > -0.024:
            bot.gripper.close()
        else: 
            bot.gripper.open()

    bot.arm.go_home()

if __name__ == "__main__":
    main()
