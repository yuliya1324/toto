## Симуляция выполнения траектории робота-манипулятора

Робот Franka Emika Panda.

Окружение: ROS Noetic, симулятор Gazebo.

Перед первым запуском:

- склонировать репозиторий и установить все модули из файла  ```.gitmodules``` по указанным адресам
- дать права на выполнение скриптов ```sudo chmod +x run-scripts/*sh```
- собрать окружение для робота** в docker-контейнер: ```sudo docker build -t sim-img . --network=host --build-arg from=ubuntu:20.04```


#### Запустить симулятор

- Запустить docker-контейнер ```sudo ./run-scripts/run_docker.sh```
- Перейти в рабочую директорию ```cd workspace```
- Собрать проект ```catkin build```
- Прописать пути ```source devel/setup.bash```
- Запустить симулятор ```roslaunch world_description panda_world.launch```


#### Запустить управление роботом

- Зайти в docker-контейнер <code>sudo docker exec -ti sim bash</code>
- Перейти в рабочую директорию ```cd workspace```
- Прописать пути ```source devel/setup.bash```
- Запустить симулятор ```roslaunch robot_control move_robot.launch```


#### Команды для управления роботом

- Управление джоинтами робота с клавиатуры ```rosrun robot_control keyboard_move_robot```
- Повернуться  ```rostopic pub /robot_rotation std_msgs/String "data: ''" --once```
- Взять ложку ```rostopic pub /robot_control/take_spoon std_msgs/String "data: ''" --once```
- Закрыть гриппер ```rostopic pub close_gripper std_msgs/String "data: ''" --once```
- Подняться ```rostopic pub /robot_up std_msgs/String "data: ''" --once```
- Повернуться в сторону тарелки ```rostopic pub /robot_move_bowl std_msgs/String "data: ''" --once```
- Опуститься ```rostopic pub /robot_down_bowl std_msgs/String "data: ''" --once```
- Зачерпнуть ```rostopic pub /robot_control/scoop_spoon std_msgs/String "data: ''" --once```


#### Прочие команды 

- Зайти в docker-контейнер <code>sudo docker exec -ti sim bash</code>
- Запустить симулятор ```roslaunch franka_gazebo panda.launch```
- Очистить сборку ```catkin clean```
- Простое раскачивание робота ```
cd /workspace/src/libs/panda_simulator/panda_simulator_examples/scripts
python3 move_robot.py```
- Отдельно запустить камеру ```roslaunch room_camera camera.launch```
- Установить робота в нейтральное положение ```rostopic pub /set_neutral_pose std_msgs/String "data: ''" --once```
- Открыть гриппер ```rostopic pub /open_gripper std_msgs/String "data: ''" --once```
- Закрыть гриппер ```rostopic pub /close_gripper std_msgs/String "data: ''" --once```


#### Сообщения для публикации

```
rostopic pub /panda_simulator/motion_controller/arm/joint_commands franka_core_msgs/JointCommand "{names: ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', panda_joint7],
position: [0, 0, 0, 0, 0, 0, 0],
velocity: [0, 0, 0, 0, 0, 0, 0],
acceleration: [0, 0, 0, 0, 0, 0, 0],
effort: [0, 0, 0, 0, 0, 0, 0]}" --once 
```

