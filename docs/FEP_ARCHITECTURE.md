# FEP (Free Energy Principle) 架构文档

本文档描述 DreamerV3-Curiosity 中 FEP 内在动机模块的设计与实现。

---

## 1. 概述

在 DreamerV3 的基础上，引入基于自由能原理 (Free Energy Principle) 的内在动机机制。核心思想：通过世界模型的信息增益 (Information Gain) 作为探索奖励，鼓励 agent 探索不确定的状态空间区域。

### 与原版 DreamerV3 的区别

| 模块 | DreamerV3 原版 | DreamerV3-Curiosity (FEP) |
|------|---------------|--------------------------|
| 奖励 | 纯外在奖励 r_ext | r_ext + beta * InfoGain |
| 网络 | 标准 heads | 新增 info_gain_head |
| 策略目标 | max E[sum r_ext] | max E[sum r_aug] |
| Imagination 梯度 | sg(imgfeat) | 移除 sg，允许梯度回流 |

---

## 2. 核心模块

### 2.1 Info Gain Head

**文件**: `dreamerv3/agent.py:70-72`

```python
self.info_gain = embodied.jax.MLPHead(
    scalar, 'mse', **config.info_gain_head, name='info_gain')
```

**配置** (`configs.yaml:100`):
```yaml
info_gain_head: {layers: 4, units: 512, act: silu, norm: layer, outscale: 0.1}
```

**功能**: 预测从当前状态出发、未来 H 步的累积信息增益。

### 2.2 Info Gain 训练目标

**文件**: `dreamerv3/agent.py:189-205`

训练目标 = 未来 H 步的累积 KL 散度 (z-score 归一化)：

```
ig_target(t) = zscore( sum_{h=0}^{H-1} KL(posterior(t+h) || prior(t+h)) )
```

其中:
- H = `config.imag_length` = 15
- KL 来自 RSSM 的 dyn_loss: `KL(q(z|h) || p(z|h))`
- z-score 归一化增强梯度信号

### 2.3 奖励增强

**文件**: `dreamerv3/agent.py:232-263`

```
img_info_gain = info_gain_head(feat)           # 预测信息增益
info_gain_norm = clip(img_info_gain / 2, -1, 1) # 归一化到 [-1, 1]
efe_bonus = effective_beta * info_gain_norm     # 内在奖励
augmented_rew = clip(img_rew + efe_bonus, 0, 1) # 混合奖励
```

---

## 3. 层级约束机制 (Performance Gate)

### 3.1 动机

固定 beta 的问题：agent 在"赢球策略"和"探索策略"之间反复摆动，出现周期性的策略崩塌。根本原因是内在奖励导致的非平稳 MDP——当世界模型对赢球状态预测准确后，InfoGain 下降，策略被引导到"新奇但无用"的状态。

灵感来自人脑的层级目标结构：前额叶皮层维持高层目标（赢球），好奇心驱动的探索受高层目标约束，不能损害核心任务表现。

### 3.2 设计

追踪 replay buffer 中真实环境 reward 的 EMA 及其历史峰值（带衰减），当性能从峰值下降时自动抑制探索 bonus：

```
perf_ema(t) = (1 - rate) * perf_ema(t-1) + rate * mean(obs['reward'])
decayed_peak = (1 - peak_decay) * perf_peak(t-1) + peak_decay * perf_ema(t)
perf_peak(t) = max(decayed_peak, perf_ema(t))
perf_drop = max(perf_peak - perf_ema, 0)
perf_gate = exp(-gate_scale * perf_drop)        # 范围 [0, 1]
effective_beta = beta * perf_gate
```

信号源选择：使用真实环境 reward（`obs['reward']`）而非 imagination reward，避免世界模型预测误差导致 gate 误判。

Peak 衰减：peak 缓慢向 ema 靠拢（`peak_decay=0.005`），防止一次偶然高分永久锁死 gate。如果 agent 持续表现低于 peak，peak 会逐渐下调，gate 重新打开。

### 3.3 行为

```
性能稳步上升:  perf_ema ~ perf_peak -> drop ~ 0   -> gate ~ 1   -> 正常探索
性能开始下跌:  perf_ema < peak      -> drop > 0   -> gate 下降   -> 探索被抑制
性能崩塌:      drop 很大            -> gate ~ 0   -> 纯外在奖励  -> 全力恢复
性能恢复后:    perf_ema 回升        -> drop 缩小  -> gate 回升   -> 重新允许探索
```

### 3.4 实现

**状态变量** (`agent.py:74-76`):
```python
self.perf_ema = nj.Variable(jnp.zeros, (), f32, name='perf_ema')
self.perf_peak = nj.Variable(jnp.zeros, (), f32, name='perf_peak')
```

**配置** (`configs.yaml:105`):
```yaml
fep: {beta: 0.05, gate_scale: 100.0, perf_ema_rate: 0.01, peak_decay: 0.005}
```

**参数说明**:
- `beta`: 内在奖励权重上界
- `gate_scale`: 性能下降时的抑制敏感度。gate_scale=100 时，drop=0.01 -> gate=0.37，drop=0.05 -> gate=0.007
- `perf_ema_rate`: EMA 更新速率，越小追踪越平滑
- `peak_decay`: peak 向 ema 靠拢的速率，越大 peak 下降越快（0.005 ≈ 200 步半衰期）

### 3.5 监控指标

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `fep/perf_ema` | 外在奖励 EMA | 应随训练上升 |
| `fep/perf_peak` | 历史峰值 | 单调递增 |
| `fep/perf_gate` | Gate 开度 | 0~1，接近 1 表示正常探索 |
| `fep/effective_beta` | 实际 beta | = beta * gate |

---

## 4. 信号流图

```
Replay Buffer Data
       |
       v
  [Encoder + RSSM]
       |
       v
  repfeat (posterior states)
       |
       |---> [Info Gain Head 训练]
       |      ig_target = zscore(cumulative_KL)
       |      loss = MSE(info_gain_head(feat), ig_target)
       |
       v
  [Imagination (H=15 steps)]
       |
       v
  imgfeat (imagined states)
       |
       |---> info_gain_head(imgfeat) ---> info_gain_norm
       |                                       |
       |---> rew_head(imgfeat) ---> img_rew ---+
       |                              |        |
       |                    [Performance Gate]  |
       |                      perf_ema, peak   |
       |                              |        |
       |                      effective_beta   |
       |                              |        |
       |                      efe_bonus = effective_beta
       |                                * info_gain_norm
       |                                       |
       |                      augmented_rew = img_rew
       |                                + efe_bonus
       |                                       |
       v                                       v
  [Policy + Value Training]
       policy_loss = -logpi * advantage(augmented_rew)
       value_loss = MSE(value, lambda_return)
```

---

## 5. 配置参考

### 完整 FEP 相关配置

```yaml
agent:
  loss_scales:
    info_gain: 1.0               # info_gain_head 训练损失权重

  info_gain_head:
    layers: 4                    # MLP 层数
    units: 512                   # 每层宽度
    act: silu
    norm: layer
    outscale: 0.1

  fep:
    beta: 0.05                   # 内在奖励权重上界
    gate_scale: 100.0            # 性能下降抑制敏感度
    perf_ema_rate: 0.01          # 性能追踪 EMA 速率
    peak_decay: 0.005            # peak 衰减速率
```

### 运行示例

```bash
# FEP + Performance Gate (默认配置)
python dreamerv3/main.py \
  --configs atari \
  --task atari_pong \
  --run.train_ratio 32 \
  --run.steps 4000000 \
  --logdir ./runs/fep_gate/pong

# 调整 gate 灵敏度
python dreamerv3/main.py \
  --configs atari \
  --task atari_pong \
  --run.train_ratio 32 \
  --run.steps 4000000 \
  --agent.fep.gate_scale 50.0 \
  --logdir ./runs/fep_gate/pong_gs50
```

---

## 6. 计划中的扩展

### 6.1 精度加权 (Precision Weighting)

让 beta 成为关于状态的函数，用 RSSM 先验分布的 entropy 度量：

```
prior_ent = entropy(p(z|h))
precision_weight = normalize(prior_ent)         # [0, 1] per state
effective_beta = beta * precision_weight * ...
```

世界模型对当前状态已确定时 (prior_ent 低) beta -> 0，不确定时 beta 较大。

### 6.2 习惯化 (Habituation)

对 info_gain 做 EMA baseline 扣除，只有超出"习以为常"水平的 info_gain 才给 bonus：

```
ig_ema = EMA(info_gain_norm.mean())
ig_surprise = max(info_gain_norm - ig_ema, 0)
efe_bonus = beta * ig_surprise
```

### 6.3 三机制组合

```
efe_bonus = beta_max * precision_weight * ig_surprise * perf_gate
```

---

## 7. 变更历史

| 日期 | 变更 | 文件 |
|------|------|------|
| 2026-03-08 | 初版 FEP 实现：info_gain_head + 固定 beta 奖励增强 | agent.py, configs.yaml |
| 2026-03-10 | 添加层级约束 (Performance Gate) 机制 | agent.py, configs.yaml |
| 2026-03-17 | Gate 信号源改为真实 reward + peak 衰减机制 | agent.py, configs.yaml |
