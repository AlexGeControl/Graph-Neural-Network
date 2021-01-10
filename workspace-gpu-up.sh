#!/bin/bash

docker run \
  --detach \
  --gpus all \
  --privileged \
  -v ${PWD}/workspace:/workspace \
  -v ${PWD}/docker/environment:/opt/environment \
  -p 49001:9001 \
  -p 45901:5901 \
  -p 40080:80 \
  -p 46006:6006 \
  -e VNC_PASSWORD=sensorfusion \
  --name x-vectornet-workspace-gpu x-vectornet-workspace:bionic-gpu-vnc