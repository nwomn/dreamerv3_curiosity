# DreamerV3 + FEP 内在动机探索

## 项目结构
- `dreamerv3/agent.py` — 核心 agent，FEP 逻辑在 loss() 函数中 (~165-280行)
- `dreamerv3/configs.yaml` — 配置文件，FEP 参数在 ~105 行
- `embodied/run/train.py` — 训练循环
- `embodied/core/replay.py` — Replay buffer
- `docs/FEP_ARCHITECTURE.md` — FEP 架构文档

## 环境
- Python: `/9950backfile/liguoqi/brainvlm/rzq/.conda/envs/myenv/bin/python`
- 启动训练必须用 `CUDA_VISIBLE_DEVICES` 隔离 GPU，否则 JAX 会在所有 GPU 上分配显存
- Montezuma ROM 任务名: `atari_montezuma_revenge`

## 分支
- `fep-true-gradient` — 主开发分支
- `fep-gate-v1` — Performance Gate v1 快照（imagination reward + 固定 peak）

## Gate 版本演进
- v1 (`fep-gate-v1`): imagination reward + 固定 peak, gate_scale=100
  - Pong: 比无gate好(消除崩塌周期), 但不如baseline
  - Montezuma: 全阶段优于baseline, 但gate被永久锁死(gate=0)
  - 问题: gate_scale与reward尺度耦合; peak只升不降被spike锁死
- v2 (当前): imagination reward + peak 衰减(peak_decay=0.005)
  - 尝试过真实obs['reward']作为信号源，但Montezuma reward太稀疏(全0)导致gate无信号，已回退
  - 待验证: peak衰减是否能解决gate永久锁死问题
  - 待验证: gate_scale自适应(相对drop归一化)

## 已知问题
- gate_scale=100 在不同 reward 尺度环境间不通用(Pong ~0.01 vs Montezuma ~1.0)
- 真实 env reward 不适合做 gate 信号源(稀疏环境中 replay batch 几乎全是 0)
- Montezuma 对随机种子极度敏感，单次实验不足以下结论

## 训练启动
```bash
cd /9950backfile/liguoqi/brainvlm/rzq/dreamerv3_curiosity
CUDA_VISIBLE_DEVICES=X nohup /9950backfile/liguoqi/brainvlm/rzq/.conda/envs/myenv/bin/python -u dreamerv3/main.py \
  --configs atari --task atari_GAME \
  --run.train_ratio 32 --run.steps 4000000 \
  --jax.train_devices 0 --jax.policy_devices 0 \
  --logdir ./runs/EXPERIMENT_NAME \
  >> ./runs/LOG_NAME.log 2>&1 &
```

## 注意事项
- 日志用 `>>` 追加，不要覆盖
- 启动前先 `nvidia-smi` 检查空闲 GPU
- 重要版本用 `git branch` 保存快照
- Montezuma 启动时 "Replay buffer empty" 等待 ~5 分钟属正常
