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


    # def room_camera_depth(self, msg):
    #     try:
    #         cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    #         self.ImageRoomDepth = cv_image
    #     except CvBridgeError, e:
    #         rospy.logerr("CvBridge Error: {0}".format(e))
    

        # if self.Image1 is not None and self.Image3 is not None:
        #     img1 = cv2.cvtColor(self.Image1, cv2.COLOR_BGR2GRAY)
        #     img2 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        #     minDisparity = 0
        #     numDisparities = 14

        #     stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        #     disparity = stereo.compute(img1, img2)
        #     disparity = disparity.astype(np.float32)
        #     disparity = (disparity/16.0 - minDisparity)/numDisparities

        #     self.Image3 = disparity
        # else:
        #     self.Image3 = cv_image


    def spin(self):

        start_time = time.time()
        while not rospy.is_shutdown():
            self.rate.sleep()

            if self.ImageRoom is not None and self.ImageRoomDepth is not None:
                cv2.imshow("Room camera", self.ImageRoom)
                cv2.imshow("Room camera depth", self.ImageRoomDepth)
                cv2.waitKey(3)

            # if self.ImageRoomDepth is not None:
            #     img = self.cv_bridge.cv2_to_imgmsg(self.ImageRoomDepth)

            #     _, buffer_img= cv2.imencode('.jpg', self.ImageRoomDepth)

            #     self.msg_img.data = base64.b64encode(buffer_img).decode("utf-8")
            #     self.msg_img.encoding = 'base64'
            #     self.msg_img.width = img.width
            #     self.msg_img.height = img.height

            #     self.pub.publish(self.msg_img)


    def shutdown(self):
        rospy.sleep(1)


camera = Camera()

camera.spin()
