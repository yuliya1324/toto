// Считывание нажатий клавиш на клавиатуре
//=====rosrun armbot_move moveMotor
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <iostream>
#include <string>
#include <signal.h>
#include <termios.h>
#include <stdio.h>

#define FORWARD 65
#define INVERSE 66
#define RESET 48

// Направление движения двигателя, отправляется на плате Arduino
#define FORWARD_DIRECTION 1
#define INVERSE_DIRECTION -1

int getch() {
  static struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);           // save old settings
  newt = oldt;
  newt.c_lflag &= ~(ICANON);                 // disable buffering
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);  // apply new settings

  int c = getchar();  // read character (non-blocking)

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // restore old settings
  return c;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "move");

    ros::NodeHandle n;
    ros::Publisher joint_move = n.advertise<std_msgs::String>("move_joint", 1000);

    int jointIndex = 0;

    ROS_INFO("Select the joint (click on the number): from 1 to 7\n0) Reset (if joint has been selected)");

    while (ros::ok()) {
        int c = getch();   // call your non-blocking input function

        std::cout << "char: " << c << std::endl;
        double joint_val;

        std_msgs::String msg;
        std::stringstream msgText;

        if (jointIndex == 0) {
            switch (c) {
                case '1':
                    jointIndex = 1;
                    break;
                case '2':
                    jointIndex = 2;
                    break;
                case '3':
                    jointIndex = 3;
                    break;
                case '4':
                    jointIndex = 4;
                    break;
                case '5':
                    jointIndex = 5;
                    break;
                case '6':
                    jointIndex = 6;
                    break;
                case '7':
                    jointIndex = 7;
                    break;    
                default:
                    ROS_ERROR("Invalid button pressed. Please click 1, 2, 3, 4, 5, 6, 7");
            }

            if (jointIndex == 0) {
                ROS_INFO("No joint selected");
            } else {
                ROS_INFO("Joint selected: %d, put ↑ (forward) or ↓ (inverse) for move joint.", jointIndex);
            }
        } else {
            switch (c) {
                case FORWARD:
                    std::cout << "Input joint value: ";
                    std::cin >> joint_val;
                    msgText << jointIndex << ":" << FORWARD_DIRECTION * joint_val;
                    msg.data = msgText.str();
                    joint_move.publish(msg);
                    break;
                case INVERSE:
                    std::cout << "Input joint value: ";
                    std::cin >> joint_val;
                    msgText << jointIndex << ":" << INVERSE_DIRECTION * joint_val;
                    msg.data = msgText.str();
                    joint_move.publish(msg);
                    break;
                case RESET:
                   jointIndex = 0;
                   ROS_INFO("Joint was reset.\nSelect the joint (click on the number): from 1 to 7\n0) Reset (if joint has been selected)");
                   break;
            }
        }
    }

    ros::spin();

    return 0;
}