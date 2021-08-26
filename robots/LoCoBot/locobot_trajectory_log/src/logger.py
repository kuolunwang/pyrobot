#!/usr/bin/env python

import rospy
import rospkg
import os
import time
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import csv

class Collect(object):
    def __init__(self, number):

        # parameter
        self.number = number
        self.trigger = None
        self.grasped_info = None
        self.traj_info = None
        self.rgb = None
        self.depth = None
        self.nu = 1

        self.cv_bridge = CvBridge()
        r = rospkg.RosPack()
        self.path = os.path.join(r.get_path('locobot_trajectory_log'), "log")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # ros service
        start = rospy.Service("/start_collect", Trigger, self.start)
        stop = rospy.Service("/stop_collect", Trigger, self.stop)

        # ros subscriber
        img_rgb = message_filters.Subscriber('/camera/color/image_raw', Image)
        img_depth = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([img_rgb, img_depth], 5, 5)
        self.ts.registerCallback(self.register)

        traj = rospy.Subscriber('/joint_states', JointState, self.traj_callback)
        grasped = rospy.Subscriber('/gripper/state', Int8, self.grasped_callback)

        # save data
        self.save()

    def start(self, req):

        res = TriggerResponse()

        try:
            self.trigger = True
            res.success = True
            self.number += 1
            self.nu = 1
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def stop(self, req):

        res = TriggerResponse()

        try:
            self.trigger = False
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def grasped_callback(self, msg):

        self.grasped_info = msg.data

    def traj_callback(self, msg):

        self.traj_info = msg.position[0:7]
        
    def writer_traj_csv(self, path, file_name, data, ti):

        dic = {}
        joint = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        for i in range(7):
            dic[joint[i]] = data[i]
        dic['timestamp'] = ti
        joint.append('timestamp')
        self.traj_list = []
        self.traj_list.append(dic)
        
        with open(os.path.join(path, file_name + '.csv'), 'a') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames = joint)
            if self.nu == 1:
                writer.writeheader()
            writer.writerows(self.traj_list)

    def writer_gra_csv(self, path, file_name, data, ti):

        dic = {}
        title = ['grasped_info', 'timestamp']
        dic['timestamp'] = ti
        dic['grasped_info'] = data
        self.grasped_list = []
        self.grasped_list.append(dic)
        
        with open(os.path.join(path, file_name + '.csv'), 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = title)
            if self.nu == 1:
                writer.writeheader()
            writer.writerows(self.grasped_list)

    def register(self, rgb, depth):

        self.rgb = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.depth = self.cv_bridge.imgmsg_to_cv2(depth, "16UC1")
        self.depth = np.array(self.depth) / 1000.0

    def save(self):

        while True:

            if self.trigger:

                rospy.loginfo('Start collect data!')

                log_path = os.path.join(self.path, "log_{:03}".format(self.number))
                img_path = os.path.join(log_path, "img")
                dep_path = os.path.join(log_path, "dep")

                if not os.path.exists(log_path):
                    os.makedirs(log_path)

                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                if not os.path.exists(dep_path):
                    os.makedirs(dep_path)

                ti = time.time()
                timestamp = str(ti)
                time.sleep(0.1)

                self.writer_traj_csv(log_path, "trajectory_info", self.traj_info, timestamp)
                self.writer_gra_csv(log_path, "grasped_info", self.grasped_info, timestamp)

                img_name = os.path.join(img_path, timestamp + "_img.jpg")
                depth_name = os.path.join(dep_path, timestamp + "_dep.npy")
                
                cv2.imwrite(img_name, self.rgb)
                np.save(depth_name, self.depth)
                self.nu += 1

if __name__ == "__main__":

    rospy.init_node("collect_data_node")

    number = rospy.get_param("number")
    collecter = Collect(number)
    rospy.spin()
