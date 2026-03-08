#!/bin/bash

# 新版 FEP 在 Atari Pong 上的测试
# GPU: 3
# 预计时间: 1-2 小时

export CUDA_VISIBLE_DEVICES=3

python dreamerv3/main.py \
  --configs atari \
  --task atari_pong \
  --agent.fep.enabled True \
  --logdir ./runs/fep_atari/pong \
  --run.steps 400000 \
  2>&1 | tee ./runs/fep_atari_pong.log

