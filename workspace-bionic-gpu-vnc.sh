#!/bin/bash

docker run \
  --detach \
  --gpus all \
  --privileged \
  -v ${PWD}/workspace:/workspace \
  -v ${PWD}/docker/environment:/opt/environment \
  -v ${PWD}/docker/image/startup.sh:/startup.sh \
  -p 40022:22 \
  -p 49001:9001 \
  -p 45901:5901 \
  -p 40080:80 \
  -p 46006:6006 \
  -e VNC_PASSWORD=deeplearning \
  --name workspace-bionic-gpu-vnc deep-learning-workspace:bionic-gpu-vnc