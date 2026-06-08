# RoboDeploy Docker images

## CPU (MuJoCo + dev tooling)

```bash
docker build -f docker/Dockerfile.cpu -t robodeploy/robodeploy:cpu .
docker run --rm robodeploy/robodeploy:cpu robodeploy run-episode --dummy --steps 10
```

## Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

## GPU (MuJoCo + torch)

```bash
docker build -f docker/Dockerfile.gpu -t robodeploy/robodeploy:gpu .
```

## Isaac Sim (NVIDIA base, ~10 GB)

```bash
docker build -f docker/Dockerfile.isaacsim -t robodeploy/robodeploy:isaacsim .
# Run with Isaac's python:
docker run --gpus all --rm robodeploy/robodeploy:isaacsim \
  /isaac-sim/python.sh -m robodeploy.cli run-episode --dummy --steps 10
```

## ROS 2 Jazzy + ros_gz_bridge

```bash
docker build -f docker/Dockerfile.ros2 -t robodeploy/robodeploy:ros2 .
docker run --rm robodeploy/robodeploy:ros2 bash -lc \
  'source /opt/ros/jazzy/setup.bash && robodeploy --help'
```

## Compose profiles

```bash
docker compose -f docker/docker-compose.yml --profile gpu build
docker compose -f docker/docker-compose.yml --profile ros2 build
```
