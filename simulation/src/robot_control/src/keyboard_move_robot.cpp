// Считывание нажатий клавиш на клавиатуре
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
    ros::Rate loop_rate(10);

      while (ros::ok()) {
        double joint_val;
        int joint_id;

        std::cout << "Input joint_id (from 1 to 7): " << std::endl;
        std::cin >> joint_id;

        if (joint_id < 1 && joint_id > 7) {
            std::cout << "Invalid value jpint_id";
            continue;
        } 

        std::cout << "Input joint value (0.01 - 1): " << std::endl;
        std::cin >> joint_val;

        if (joint_val < -1 && joint_val > 1) {
            std::cout << "Invalid value joint";
            continue;
        } 

        std_msgs::String msg;
        std::stringstream msgText;

        msgText << joint_id << ":" << joint_val;
        msg.data = msgText.str();
        joint_move.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::spin();

    return 0;
}