#!/usr/bin/env python3
import rospy
import time
from sensor_msgs.msg import JointState
from franka_core_msgs.msg import JointCommand, RobotState
from std_msgs.msg import String
import numpy as np
from copy import deepcopy
from panda_robot import PandaArm

class RobotMove():

    def __init__(self):
        rospy.init_node("move_robot")
        rospy.wait_for_service('/controller_manager/list_controllers')
        rospy.loginfo("Starting node...")
        rospy.sleep(5)

        self.vals = []
        self.vels = []
        self.names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']

        self.neutral_pose = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
        self.test_pose1 = [-1.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
        self.test_pose2 = [1.117792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
        
        # self.pose_pick_spoon = [1.2592927325038223, 0.3072768833743007, 0.14392324019225633, -2.747312906392179, 0.21969104115050886, 2.9596374531242233, 0.8542973732233126]
        self.pose_pick_spoon = [1.2694259100931253, 0.5017555223461567, 0.14455989557824633, -2.8514408819243116, 0.217828597411871, 3.721783509160593, 0.8513885413857256]
        self.pose_pick_up = [1.2794313855805983, -0.19364285432803596, 0.16087728058255735, -2.5004812253647417, 0.21287684625265246, 2.1301047033423677, 1.348273183424749]
        self.pose_move_bowl = [0.5794222336742347, -0.19364285432803596, 0.16087728058255735, -2.5004812253647417, 0.21287684625265246, 2.1301047033423677, 1.348273183424749]
        self.pose_bowl_down = [0.5795271120171588, -0.004413267742999327, 0.166347361280617, -2.643042301642605, 0.21003053822046613, 2.511649762270033, 1.3481553572775278]
        self.pose_scoop_spoon = [0.5763586362343149, -0.2854728769361219, 0.16343975617146178, -2.647924737655005, 0.2094620096840556, 2.552383418763543, 1.347922506764509]

        self.step = 0.01
        self.t = 0
        self.initial_pose = self.vals
        self.rate = rospy.Rate(1000)
        self.robot = PandaArm()
        self.robot.set_joint_position_speed(0.1)


        self.pub_joints = rospy.Publisher('/panda_simulator/motion_controller/arm/joint_commands',JointCommand, queue_size = 1, tcp_nodelay = True)

        self.pub_take_spoon = rospy.Publisher('/robot_control/take_spoon', String, queue_size = 1)
        self.pub_scoop_spoon = rospy.Publisher('/robot_control/scoop_spoon', String, queue_size = 1)

        # Subscribe to robot joint state
        rospy.Subscriber('/panda_simulator/custom_franka_state_controller/joint_states', JointState, self.callback)

        # Take spoon 
        rospy.Subscriber('/robot_control/take_spoon', String, self.take_spoon)

        # Scoop with spoon
        rospy.Subscriber('/robot_control/scoop_spoon', String, self.scoop_spoon)

        # Move joint
        rospy.Subscriber('/move_joint', String, self.move_joint)

        # Set to neutral_pose
        rospy.Subscriber('/set_neutral_pose', String, self.set_neutral_pose)

        # Open gripper
        rospy.Subscriber('/open_gripper', String, self.open_gripper)

        # Close gripper
        rospy.Subscriber('/close_gripper', String, self.close_gripper)

        # Go to right
        rospy.Subscriber('/robot_rotation', String, self.robot_rotation)

        # Go up
        rospy.Subscriber('/robot_up', String, self.robot_up)
        rospy.Subscriber('/robot_move_bowl', String, self.robot_move_bowl)
        rospy.Subscriber('/robot_down_bowl', String, self.robot_down_bowl)

        # Subscribe to robot state (Refer JointState.msg to find all available data. 
        # Note: All msg fields are not populated when using the simulated environment)
        rospy.Subscriber('/panda_simulator/custom_franka_state_controller/robot_state', RobotState, self.state_callback)

        rospy.loginfo("Not recieved current joint state msg")

        while not rospy.is_shutdown() and len(self.vals) != 7:
            continue

        rospy.loginfo("Sending robot to neutral pose")
        self.send_to_neutral() # DON'T DO THIS ON REAL ROBOT!! (use move_to_neutral() method from ArmInterface of franka_ros_interface package)

        # Open gripper
        self.robot.get_gripper().open()

        rospy.sleep(2.0)

        rospy.loginfo("Commanding...\n")
        rospy.on_shutdown(self.shutdown)


    def callback(self, msg):

        temp_vals = []
        temp_vels = []
        for n in self.names:
            idx = msg.name.index(n)
            temp_vals.append(msg.position[idx])
            temp_vels.append(msg.velocity[idx])

        self.vals = deepcopy(temp_vals)
        self.vels = deepcopy(temp_vels)


    def take_spoon(self, msg):
        self.robot.move_to_joint_position(self.pose_pick_spoon)


    def scoop_spoon(self, msg):
        self.robot.move_to_joint_position(self.pose_scoop_spoon)


    def set_neutral_pose(self, msg):
        self.robot.move_to_joint_position(self.neutral_pose)


    def open_gripper(self, msg):
        self.robot.get_gripper().open()


    def close_gripper(self, msg):
        self.robot.get_gripper().close()


    def robot_rotation(self, msg):
        current_val = deepcopy(self.vals)
        current_val[0] += 1
        self.robot.move_to_joint_position(current_val)

    def robot_up(self, msg):
        self.robot.move_to_joint_position(self.pose_pick_up)

    def robot_move_bowl(self, msg):
        self.robot.move_to_joint_position(self.pose_move_bowl)

    def robot_down_bowl(self, msg):
        self.robot.move_to_joint_position(self.pose_bowl_down)

    def move_joint(self, msg):
        joints_pos = deepcopy(self.vals)

        [joint_id, step] = list(msg.data.split(':'))
        joints_pos[int(joint_id)-1] += float(step)

        self.robot.move_to_joint_position(joints_pos)


    def state_callback(self, msg):
        # global t
        if self.t%100 == 0:
            self.t = 1
        self.t+=1

    def send_to_neutral(self):
        # temp_pub = rospy.Publisher('/panda_simulator/motion_controller/arm/joint_commands',JointCommand, queue_size = 1, tcp_nodelay = True)
        # Create JointCommand message to publish commands
        pubmsg = JointCommand()
        pubmsg.names = self.names # names of joints (has to be 7)
        pubmsg.position = self.neutral_pose # JointCommand msg has other fields (velocities, efforts) for
                               # when controlling in other control mode
        pubmsg.mode = pubmsg.POSITION_MODE # Specify control mode (POSITION_MODE, VELOCITY_MODE, IMPEDANCE_MODE (not available in sim), TORQUE_MODE)
        curr_val = deepcopy(self.vals)

        print(curr_val)

        while all(abs(self.neutral_pose[i]-curr_val[i]) > 0.01 for i in range(len(curr_val))):
            self.pub_joints.publish(pubmsg)
            curr_val = deepcopy(self.vals)


    def spin(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            self.rate.sleep()


    def shutdown(self):
        rospy.sleep(1)


robotMove = RobotMove()

robotMove.spin()