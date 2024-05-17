#!/usr/bin/env bash

cd /home/yara/camera_ws/src/graspnet-baseline
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path /home/yara/camera_ws/src/graspnet-baseline/checkpoint-rs.tar