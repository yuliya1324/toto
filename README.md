# Repository for TOTO benchmark solution
<!-- TODO: add teaser figures, some setup/task images, etc  -->
![toto_dataset](docs/images/toto_dataset.gif)

## Prerequisites
- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)

## Installation
You can either use a local conda environment or a docker environment.

### Setup conda environment
1. Run the following command to create a new conda environment: ```source setup_toto_env.sh```

### Setup docker environment
1. Follow the instructions in [here](https://github.com/AGI-Labs/toto_benchmark/blob/main/docker/README.md).

Note: If you are contributing models to TOTO, we strongly suggest setting up the docker environment.

### TOTO Datasets
<!-- TODO: need to update the dataset link after google drive clean up -->
TOTO consists of two tabletop manipulations tasks, scooping and pouring. The datasets of the two tasks can be downloaded [here](https://drive.google.com/drive/folders/1JGPGjCqUP4nUOAxY3Fpx3PjUQ_loo7fc?usp=share_link).

*Update*: please download the scooping data from Google Cloud Bucket [here](https://console.cloud.google.com/storage/browser/toto-dataset) instead.

<!-- TODO: update link to dataset README.md file. May consider create a dataset/ folder and add the readme into the repo -->
We release the following datasets: 
- `cloud-dataset-scooping.zip`: TOTO scooping dataset
- `cloud-dataset-pouring.zip`: TOTO pouring dataset

Additional Info:
- `scooping_parsed_with_embeddings_moco_conv5.pkl`: the same scooping dataset parsed with MOCO (Ours) pre-trained visual representations. (included as part of the TOTO scooping dataset) 
- `pouring_parsed_with_embeddings_moco_conv5.pkl`: the same pouring dataset parsed with MOCO (Ours) pre-trained visual representations. 
(included as part of the TOTO pouring dataset)

For more detailed dataset format information, see `assets/README.md`

## Solutions

We present 3 models:
- Decision Transformer
- Recurrent Memory Decision Transformer
- Masked Trajectory Model

### DT

### RMDT

[RMDT](https://arxiv.org/pdf/2306.09459.pdf) is based on [DT](https://arxiv.org/pdf/2106.01345.pdf), but it has additional memory tokens as described in [RMT](https://arxiv.org/pdf/2207.06881.pdf) to process long sequences.

![model-architecture](docs/images/model.PNG)

Here's an example command to train RMDT on pouring dataset with precomputed embeddings for RGB images by pretrained MOCO as the image encoder. To add depth images set `cameras: ["cam0d"]` in config.

```
python toto_benchmark/scripts/train.py --config-name train_rmt_d.yaml data.pickle_fn=assets/cloud-data-pooring/pooring_parsed_with_embeddings_moco_conv5_robocloud.pkl
```

Here's an example command to test RMDT in a dummy environement.

```
python toto_benchmark/scripts/test_stub_env.py -f=toto_benchmark/outputs/collaborator_agent
```

### MTM

[MTM](https://arxiv.org/pdf/2305.02968.pdf) uses masking in the input sequence and reconstructs the full original sequence. This way it gets more general knowledge of the world, can perform different tasks and performs better than DT on HalfCheetah, Hopper, Walker2d environments. 

To start training on TOTO benchmark run toto_train.ipynb

## Симуляция выполнения траектории робота-манипулятора

Робот Franka Emika Panda.

Окружение: ROS Noetic, симулятор Gazebo.

Симуляция лежит в папке ```cd simulation```

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
