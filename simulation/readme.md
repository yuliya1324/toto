## Симуляция выполнения траектории робота-манипулятора

Робот Franka Emika Panda.

Окружение: ROS Noetic, симулятор Gazebo.

Перед первым запуском:

- дать права на выполнение скриптов ```sudo chmod +x run-scripts/*sh```
- собрать окружение для робота** в docker-контейнер: ```sudo docker build -t sim-img . --network=host --build-arg from=ubuntu:20.04```


#### Запустить симулятор

- Запустить docker-контейнер ```sudo ./run-scripts/run_docker.sh```.
- Перейти в рабочую директорию ```cd workspace```
- Собрать проект ```catkin build```
- Прописать пути ```source devel/setup.bash```


#### Прочие команды 

- Зайти в docker-контейнер <code>sudo docker exec -ti sim bash</code>
- Запустить симулятор ```roslaunch franka_gazebo panda.launch```
- Очистить сборку ```catkin clean```


#### Сообщения для публикации

```
rostopic pub /panda_simulator/motion_controller/arm/joint_commands franka_core_msgs/JointCommand "{names: ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', panda_joint7]
position: [0, 0, 0, 0, 0, 0, 0]
velocity: [0, 0, 0, 0, 0, 0, 0]
acceleration: [0, 0, 0, 0, 0, 0, 0]
effort: [0, 0, 0, 0, 0, 0, 0]}" --once 
```



<pre><code>rostopic pub move_robot_delay_gripper ur5_info/MoveUR5WithGripper "{positions: [position:[1.606798,-3.091649,2.827192,-1.962667,-1.436540,-0.000551]], delay: [0], gripperAngle: 0.0}" --once</code></pre>

