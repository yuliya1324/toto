## Симуляция выполнения траектории робота-манипулятора

Робот Franka Emika Panda.

Окружение: ROS Noetic, симулятор Gazebo.

Перед первым запуском:

- дать права на выполнение скриптов ```sudo chmod +x run-scripts/*sh```
- собрать окружение для робота** в docker-контейнер: ```sudo docker build -t trajopt-img . --network=host --build-arg from=ubuntu:20.04```


#### Запустить симулятор

- Запустить docker-контейнер ```sudo ./run-scripts/run_docker.sh```.
- Перейти в рабочую директорию ```cd workspace```
- Собрать проект ```catkin build```
- Прописать пути ```source devel/setup.bash```










