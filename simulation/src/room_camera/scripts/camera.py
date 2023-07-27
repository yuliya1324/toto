#!/usr/bin/env python3

import time
from math import sin, cos, sqrt, atan
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import cv2
import base64
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from room_camera.msg import ImageCamera

class Camera():

    def __init__(self):
        rospy.init_node('camera', anonymous=True)

        self.cv_bridge = CvBridge()
        self.ImageRoom = None
        self.ImageRoomDepth = None
        self.msg_img = ImageCamera()

        rospy.Subscriber("/robot/camera_room/image_raw", Image, self.room_camera)

        self.rate = rospy.Rate(30)

        rospy.on_shutdown(self.shutdown)


    def room_camera(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.ImageRoom = cv_image
            self.ImageRoomDepth = cv_image
        except (CvBridgeError, e):
            rospy.logerr("CvBridge Error: {0}".format(e))


    def spin(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            self.rate.sleep()

            if self.ImageRoom is not None and self.ImageRoomDepth is not None:
                cv2.imshow("Room camera", self.ImageRoom)
                cv2.imshow("Room camera depth", self.ImageRoomDepth)
                cv2.waitKey(3)


    def shutdown(self):
        rospy.sleep(1)


camera = Camera()

camera.spin()
