# FEP-Unified Dreamer 实验计划

**制定日期**: 2026-03-05
**目标会议**: ICLR 2027 (2026年9月截稿)
**预计完成**: 2026年8月

---

## 1. 实验目标

### 1.1 主要目标

证明 FEP-Unified Dreamer 在多个环境上相比 DreamerV3 baseline 有显著提升。

### 1.2 次要目标

1. 理解 FEP 模块的工作机制
2. 分析 Goal Proximity 和 Info Gain 的贡献
3. 研究 z_goal 的表示学习
4. 验证在不同环境类型上的泛化性

---

## 2. 实验环境

### 2.1 主要环境（必须）

| 环境 | 任务数 | 训练步数 | 预计时间 | 优先级 |
|------|--------|----------|----------|--------|
| **Crafter** | 1 | 1.1M | 24h | ⭐⭐⭐ 进行中 |
| **Atari-26** | 26 | 400k/game | 10-15天 | ⭐⭐⭐ 必须 |
| **DMC Vision** | 20 | 1M/task | 15-20天 | ⭐⭐⭐ 必须 |

**总计**: 47 个任务，预计 30-40 天

### 2.2 扩展环境（可选）

| 环境 | 任务数 | 训练步数 | 预计时间 | 优先级 |
|------|--------|----------|----------|--------|
| **Atari-57** | 57 | 200M/game | 2-3个月 | ⭐⭐ 如果时间允许 |
| **Procgen** | 16 | 50M/game | 1个月 | ⭐⭐ 泛化性测试 |
| **Minecraft** | 1 | 100M | 1周 | ⭐ 长期任务测试 |

---

## 3. 实验配置

### 3.1 FEP-Unified（主实验）

**配置文件**: `configs/fep_unified.yaml`

```yaml
agent:
  fep:
    enabled: True
    alpha_max: 1.0
    beta_max: 1.0
    beta_warmup: 100000
    kl_threshold: 2.0
    goal_ema_rate: 0.1
    goal_topk: 10
```

**运行命令**:
```bash
# Crafter
python dreamerv3/main.py --configs crafter --agent.fep.enabled True \
  --logdir runs/fep_crafter --run.steps 1100000

# Atari (每个游戏)
python dreamerv3/main.py --configs atari100k --task atari_pong \
  --agent.fep.enabled True --logdir runs/fep_atari/pong --run.steps 400000

# DMC (每个任务)
python dreamerv3/main.py --configs dmc_vision --task dmc_walker_walk \
  --agent.fep.enabled True --logdir runs/fep_dmc/walker_walk --run.steps 1000000
```

### 3.2 DreamerV3 Baseline（对比）

**配置**: 关闭 FEP 模块

```bash
# Crafter baseline
python dreamerv3/main.py --configs crafter --agent.fep.enabled False \
  --logdir runs/baseline_crafter --run.steps 1100000

# Atari baseline
python dreamerv3/main.py --configs atari100k --task atari_pong \
  --agent.fep.enabled False --logdir runs/baseline_atari/pong --run.steps 400000

# DMC baseline
python dreamerv3/main.py --configs dmc_vision --task dmc_walker_walk \
  --agent.fep.enabled False --logdir runs/baseline_dmc/walker_walk --run.steps 1000000
```

### 3.3 消融实验

#### 实验 A：只有 Info Gain

```yaml
agent:
  fep:
    enabled: True
    alpha_max: 1.0
    beta_max: 0.0  # 关闭 Goal Proximity
```

#### 实验 B：只有 Goal Proximity

```yaml
agent:
  fep:
    enabled: True
    alpha_max: 0.0  # 关闭 Info Gain
    beta_max: 1.0
```

#### 实验 C：不同的 β warmup

```yaml
# 快速 warmup
beta_warmup: 50000

# 慢速 warmup
beta_warmup: 200000
```

#### 实验 D：不同的 KL 阈值

```yaml
# 低阈值（更早转向利用）
kl_threshold: 1.5

# 高阈值（更长时间探索）
kl_threshold: 2.5
```

---

## 4. 详细时间表

### 阶段 1：Crafter 完成（1周）

**时间**: 2026-03-05 ~ 2026-03-12

| 日期 | 任务 | 预计时长 |
|------|------|----------|
| 03-05 | ✅ 当前训练继续（已完成 25.8%） | 20h |
| 03-06 | 监控训练，收集中期数据 | - |
| 03-07 | Crafter 训练完成，评估 | 2h |
| 03-08 | 运行 Crafter baseline | 24h |
| 03-09 | 对比分析，生成图表 | 4h |
| 03-10 | 消融实验：只有 GP | 24h |
| 03-11 | 消融实验：只有 IG | 24h |
| 03-12 | Crafter 结果整理 | 4h |

**交付物**:
- Crafter FEP vs Baseline 对比
- 消融实验结果
- 训练曲线和分析图表

### 阶段 2：Atari-26 实验（2-3周）

**时间**: 2026-03-13 ~ 2026-04-02

#### 第一批（高优先级，10个游戏）

**游戏选择**:
1. Pong（简单，快速验证）
2. Breakout（经典）
3. Alien（中等难度）
4. Seaquest（探索重要）
5. MsPacman（导航）
6. Qbert（策略）
7. SpaceInvaders（反应）
8. BeamRider（长期规划）
9. Enduro（持续控制）
10. Frostbite（复杂探索）

**并行策略**: 使用 4 个 GPU，每个 GPU 运行 2-3 个游戏

| 周 | 任务 | GPU 分配 |
|----|------|----------|
| Week 1 | FEP: Pong, Breakout, Alien, Seaquest | GPU 0-1 |
| Week 1 | FEP: MsPacman, Qbert, SpaceInvaders | GPU 2-3 |
| Week 2 | FEP: BeamRider, Enduro, Frostbite | GPU 0-1 |
| Week 2 | Baseline: 前5个游戏 | GPU 2-3 |
| Week 3 | Baseline: 后5个游戏 | GPU 0-1 |
| Week 3 | 结果分析和可视化 | - |

#### 第二批（完整26个游戏）

**时间**: 2026-04-03 ~ 2026-04-20

剩余 16 个游戏，继续并行训练。

**交付物**:
- Atari-26 完整结果
- 性能对比表格
- 学习曲线图

### 阶段 3：DMC Vision 实验（2-3周）

**时间**: 2026-04-21 ~ 2026-05-10

#### DMC 任务列表（20个）

**按难度分组**:

**简单** (6个):
- cartpole_swingup
- reacher_easy
- finger_spin
- cheetah_run
- walker_walk
- hopper_stand

**中等** (8个):
- walker_run
- hopper_hop
- quadruped_walk
- quadruped_run
- humanoid_stand
- humanoid_walk
- fish_swim
- acrobot_swingup

**困难** (6个):
- humanoid_run
- dog_stand
- dog_walk
- dog_run
- manipulator_bring_ball
- cup_catch

**并行策略**: 4 GPU × 5 任务/GPU

| 周 | 任务组 | 状态 |
|----|--------|------|
| Week 1 | FEP: 简单组 (6个) | GPU 0-1 |
| Week 1 | FEP: 中等组前4个 | GPU 2-3 |
| Week 2 | FEP: 中等组后4个 + 困难组前2个 | GPU 0-1 |
| Week 2 | FEP: 困难组后4个 | GPU 2-3 |
| Week 3 | Baseline: 全部20个 | 所有GPU |
| Week 4 | 结果分析 | - |

**交付物**:
- DMC-20 完整结果
- 与 DrQv2, CURL 等方法对比
- 不同任务类型的分析

### 阶段 4：消融实验（1周）

**时间**: 2026-05-11 ~ 2026-05-17

在 Crafter + 5个 Atari + 3个 DMC 上运行消融实验：

| 实验 | 环境数 | 预计时间 |
|------|--------|----------|
| 只有 Info Gain | 9 | 3天 |
| 只有 Goal Proximity | 9 | 3天 |
| 不同 β warmup | 3 | 1天 |

### 阶段 5：数据分析和可视化（1周）

**时间**: 2026-05-18 ~ 2026-05-24

- [ ] 汇总所有实验结果
- [ ] 生成性能对比表格
- [ ] 绘制学习曲线
- [ ] 统计显著性检验
- [ ] z_goal 演化分析
- [ ] 制作论文图表

### 阶段 6：Workshop 论文撰写（2周）

**时间**: 2026-05-25 ~ 2026-06-07

- [ ] 撰写初稿（4-6页）
- [ ] 制作图表和表格
- [ ] 内部审阅和修改
- [ ] 最终提交

**截稿日期**: 2026-06-10（NeurIPS Workshop）

### 阶段 7：补充实验和改进（2个月）

**时间**: 2026-06-10 ~ 2026-08-31

根据 workshop 反馈：
- [ ] 补充实验
- [ ] 深化理论分析
- [ ] 改进写作
- [ ] 准备 ICLR 投稿

**截稿日期**: 2026-09-25（ICLR 2027）

---

## 5. 资源需求

### 5.1 计算资源

**GPU 需求**:
- 4 × NVIDIA A100/V100 (40GB)
- 或 8 × RTX 3090/4090 (24GB)

**存储需求**:
- Checkpoint: ~50 GB/环境
- Replay buffer: ~20 GB/环境
- 日志和可视化: ~10 GB/环境
- **总计**: ~4 TB

**训练时间估算**:
- Crafter: 24h × 1 = 24h
- Atari-26: 12h × 26 × 2 (FEP+Baseline) = 624h ≈ 26天（4 GPU 并行 → 7天）
- DMC-20: 24h × 20 × 2 = 960h ≈ 40天（4 GPU 并行 → 10天）
- 消融: 200h ≈ 8天（4 GPU 并行 → 2天）
- **总计**: ~30天（4 GPU 全天候运行）

### 5.2 人力需求

**主要研究者**: 1人
- 实验设计和执行
- 代码调试
- 数据分析

**协助**:
- 论文写作指导
- 实验结果审阅

---

## 6. 评估指标

### 6.1 主要指标

**性能指标**:
- Episode return（平均、最大、最小）
- Success rate（如果适用）
- Sample efficiency（达到阈值所需步数）

**对比方式**:
- FEP-Unified vs DreamerV3 Baseline
- 相对提升百分比
- 统计显著性（t-test, p < 0.05）

### 6.2 分析指标

**FEP 模块**:
- Alpha 和 Beta 的演化
- Goal Proximity 的变化
- Info Gain 的贡献
- KL 散度的稳定性

**z_goal 分析**:
- 特征稀疏性
- 能量分布
- 与 reward 的相关性
- 随训练的演化

### 6.3 对比基准

| 方法 | Crafter | Atari-26 | DMC-20 |
|------|---------|----------|--------|
| DreamerV3 | 14.5% | - | - |
| Curious Replay | 19.4% | - | - |
| DrQv2 | - | - | SOTA |
| **FEP-Unified** | **目标: >15%** | **目标: >Baseline** | **目标: >Baseline** |

---

## 7. 风险和应对

### 7.1 技术风险

**风险 1**: 训练不稳定
- **应对**: 调整学习率、减小 α/β 上限
- **备选**: 使用更保守的 warmup

**风险 2**: 性能不如 baseline
- **应对**: 分析失败原因，调整超参数
- **备选**: 强调分析贡献而非性能提升

**风险 3**: GPU 资源不足
- **应对**: 优先完成核心环境（Crafter + 10 Atari + 5 DMC）
- **备选**: 使用云计算资源

### 7.2 时间风险

**风险**: 实验时间超出预期
- **应对**:
  - 减少环境数量（Atari 26 → 15）
  - 减少训练步数（在不影响收敛的前提下）
  - 跳过部分消融实验

**缓冲时间**: 预留 2 周应对意外

---

## 8. 检查点和里程碑

### 里程碑 1: Crafter 完成（Week 1）
- [ ] FEP-Unified 训练完成
- [ ] Baseline 对比完成
- [ ] 初步结果分析

### 里程碑 2: Atari 第一批完成（Week 4）
- [ ] 10个游戏 FEP 训练完成
- [ ] 10个游戏 Baseline 完成
- [ ] 性能对比表格

### 里程碑 3: 核心实验完成（Week 8）
- [ ] Crafter + Atari-26 + DMC-20 全部完成
- [ ] 所有对比实验完成
- [ ] 数据分析完成

### 里程碑 4: Workshop 投稿（Week 10）
- [ ] 论文初稿完成
- [ ] 图表制作完成
- [ ] 提交 NeurIPS Workshop

### 里程碑 5: ICLR 投稿（Week 20）
- [ ] 补充实验完成
- [ ] 论文完整版完成
- [ ] 提交 ICLR 2027

---

## 9. 实验追踪

### 9.1 日志记录

每个实验记录：
- 配置文件
- 训练曲线
- 最终性能
- 异常情况
- 计算时间

### 9.2 结果汇总

使用表格追踪：
```
experiments/
├── results_summary.csv
├── crafter/
│   ├── fep_unified/
│   ├── baseline/
│   └── ablations/
├── atari/
│   └── [game_name]/
└── dmc/
    └── [task_name]/
```

---

## 10. 下一步行动

### 立即行动（本周）

- [x] 制定实验计划（本文档）
- [ ] 准备 Atari 环境配置
- [ ] 准备 DMC 环境配置
- [ ] 设置实验追踪系统
- [ ] 等待 Crafter 训练完成

### 短期行动（下周）

- [ ] 启动 Crafter baseline
- [ ] 启动 Atari 第一批实验
- [ ] 开始数据分析脚本开发

---

## 11. GPU资源配置方案

**基准训练速度**: 647,059 步/天 (26,961 步/小时)
**数据来源**: Crafter 实际训练（1.1M步/1.7天）

### 11.1 方案对比

| 方案 | GPU数 | 最小实验 | 标准实验 | 完整实验 | 适用场景 |
|------|-------|----------|----------|----------|----------|
| **方案A** | 1 GPU | 25天 | 47天 | 97天 | 资源紧张 |
| **方案B** | 2 GPU | 13天 | 23天 | 49天 | 平衡方案 |
| **方案C** | 4 GPU | 7天 | 13天 | 25天 | 理想情况 |

### 11.2 一块GPU方案（串行训练）

#### 方案1A: 最小核心实验（25天）

**目标**: 快速验证，发Workshop
**任务数**: 22个（Crafter×2 + Atari-5×2 + DMC-5×2）

**执行顺序**:

```bash
# 阶段1: Crafter (3.4天)
1. Crafter FEP (1.7d) - 已完成 ✅
2. Crafter Baseline (1.7d)

# 阶段2: Atari-5 快速验证 (6.2天)
3. Montezuma FEP (0.62d)
4. Pitfall FEP (0.62d)
5. Venture FEP (0.62d)
6. PrivateEye FEP (0.62d)
7. Solaris FEP (0.62d)
8. Montezuma Baseline (0.62d)
9. Pitfall Baseline (0.62d)
10. Venture Baseline (0.62d)
11. PrivateEye Baseline (0.62d)
12. Solaris Baseline (0.62d)

# 阶段3: DMC-5 (15.5天)
13-17. DMC-5 FEP (walker_walk, cheetah_run, hopper_hop, quadruped_walk, humanoid_walk)
18-22. DMC-5 Baseline

总计: 25.0天
```

**启动命令模板**:
```bash
# Crafter Baseline
CUDA_VISIBLE_DEVICES=1 python -u dreamerv3/main.py \
  --configs crafter \
  --agent.fep.enabled False \
  --logdir runs/baseline_crafter \
  > runs/baseline_crafter.log 2>&1 &

# Atari (400k)
CUDA_VISIBLE_DEVICES=1 python -u dreamerv3/main.py \
  --configs atari100k \
  --task atari_montezuma_revenge \
  --agent.fep.enabled True \
  --run.steps 400000 \
  --logdir runs/fep_atari/montezuma \
  > runs/fep_atari_montezuma.log 2>&1 &

# DMC
CUDA_VISIBLE_DEVICES=1 python -u dreamerv3/main.py \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --run.steps 1000000 \
  --logdir runs/fep_dmc/walker_walk \
  > runs/fep_dmc_walker_walk.log 2>&1 &
```

#### 方案1B: 标准实验（47天）

**目标**: 充分实验，发会议论文
**任务数**: 42个（Crafter×2 + Atari-10×2 + DMC-10×2）

**额外任务**（在方案1A基础上）:
- Atari: +5个游戏（Pong, Breakout, Seaquest, MsPacman, Qbert）
- DMC: +5个任务（fish_swim, acrobot_swingup, humanoid_stand, dog_stand, cup_catch）

**总时间**: 47天

#### 方案1C: 完整实验（97天）

**任务数**: 94个（Crafter×2 + Atari-26×2 + DMC-20×2）
**总时间**: 97天（3.2个月）
**适用**: 时间充裕，追求最完整结果

### 11.3 两块GPU方案（并行训练）

#### 方案2A: 最小核心实验（13天）

**GPU分配策略**: 轮流使用，任务完成后立即启动下一个

**执行计划**:

| 时间段 | GPU 1 | GPU 2 | 并行收益 |
|--------|-------|-------|----------|
| Day 1-2 | Crafter FEP (完成) | Crafter Baseline | 节省1.7天 |
| Day 2-3 | Montezuma FEP | Pitfall FEP | 节省0.6天 |
| Day 3-4 | Venture FEP | PrivateEye FEP | 节省0.6天 |
| Day 4-5 | Solaris FEP | Montezuma Baseline | 节省0.6天 |
| Day 5-6 | Pitfall Baseline | Venture Baseline | 节省0.6天 |
| Day 6-7 | PrivateEye Baseline | Solaris Baseline | 节省0.6天 |
| Day 7-13 | DMC-5 FEP (轮流) | DMC-5 Baseline (轮流) | 节省7.8天 |

**总时间**: 12.6天（约13天）
**节省**: 25天 → 13天（节省48%）

**并行启动示例**:
```bash
# GPU 1: Crafter Baseline
CUDA_VISIBLE_DEVICES=1 python -u dreamerv3/main.py \
  --configs crafter --agent.fep.enabled False \
  --logdir runs/baseline_crafter &

# GPU 2: Atari Montezuma
CUDA_VISIBLE_DEVICES=2 python -u dreamerv3/main.py \
  --configs atari100k --task atari_montezuma_revenge \
  --agent.fep.enabled True --run.steps 400000 \
  --logdir runs/fep_atari/montezuma &
```

#### 方案2B: 标准实验（23天）

**任务数**: 42个
**总时间**: 23天
**节省**: 47天 → 23天（节省51%）

#### 方案2C: 完整实验（49天）

**任务数**: 94个
**总时间**: 49天
**节省**: 97天 → 49天（节省50%）

### 11.4 四块GPU方案（最优并行）

仅作参考，当前不可用。

**最小实验**: 7天
**标准实验**: 13天
**完整实验**: 25天

### 11.5 执行策略建议

#### 优先级排序

**第一优先级**（必须完成）:
1. Crafter FEP + Baseline
2. Atari-5 探索敏感游戏（Montezuma, Pitfall, Venture, PrivateEye, Solaris）
3. DMC-5 基础任务（walker_walk, cheetah_run, hopper_hop, quadruped_walk, humanoid_walk）

**第二优先级**（充实论文）:
4. Atari +5个经典游戏（Pong, Breakout, Seaquest, MsPacman, Qbert）
5. DMC +5个任务

**第三优先级**（如果时间允许）:
6. 扩展到完整Atari-26和DMC-20

#### 灵活调整原则

1. **先跑FEP，再决定Baseline**
   - 如果FEP效果不好，省下Baseline时间调整方向
   - 如果FEP效果好，再跑Baseline对比

2. **使用Atari 100k快速验证**
   - 100k步只需3.7小时，400k需14.8小时
   - 先用100k验证方法有效性
   - 确认有效后再跑400k

3. **实时监控，及时止损**
   - 每天检查训练曲线
   - 发现问题立即停止，避免浪费时间

4. **根据GPU可用性动态切换**
   - 1 GPU可用 → 执行方案1A/1B
   - 2 GPU可用 → 立即切换到方案2A/2B
   - 随时可以在方案间切换

### 11.6 时间节点规划

#### 1 GPU方案时间线

**Week 1** (3/6-3/12):
- Crafter Baseline完成
- Atari-5 FEP开始（完成2-3个）

**Week 2** (3/13-3/19):
- Atari-5 FEP完成
- Atari-5 Baseline开始

**Week 3** (3/20-3/26):
- Atari-5 Baseline完成
- DMC-5 FEP开始（完成1-2个）

**Week 4-5** (3/27-4/9):
- DMC-5 FEP完成
- DMC-5 Baseline开始

**Week 6** (4/10-4/16):
- DMC-5 Baseline完成
- 数据分析和可视化

**里程碑**: 4月中旬完成最小核心实验（25天）

#### 2 GPU方案时间线

**Week 1** (3/6-3/12):
- Crafter Baseline完成
- Atari-5 FEP全部完成
- Atari-5 Baseline开始

**Week 2** (3/13-3/19):
- Atari-5 Baseline完成
- DMC-5 FEP开始（完成3-4个）

**Week 3** (3/20-3/26):
- DMC-5 FEP完成
- DMC-5 Baseline完成大部分

**里程碑**: 3月底完成最小核心实验（13天）

### 11.7 自动化脚本

创建任务队列脚本，自动启动下一个任务：

```bash
# scripts/run_queue.sh
#!/bin/bash

QUEUE_FILE="task_queue.txt"
GPU_ID=$1

while IFS='|' read -r task_name config task args logdir; do
    echo "[$(date)] Starting: $task_name"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u dreamerv3/main.py \
        --configs $config \
        --task $task \
        $args \
        --logdir $logdir \
        > ${logdir}.log 2>&1

    echo "[$(date)] Completed: $task_name"

    # 发送通知（可选）
    # echo "$task_name completed" | mail -s "Training Update" your@email.com

done < "$QUEUE_FILE"

echo "[$(date)] All tasks completed!"
```

**任务队列文件示例** (`task_queue.txt`):
```
Crafter Baseline|crafter||--agent.fep.enabled False|runs/baseline_crafter
Montezuma FEP|atari100k|atari_montezuma_revenge|--agent.fep.enabled True --run.steps 400000|runs/fep_atari/montezuma
Pitfall FEP|atari100k|atari_pitfall|--agent.fep.enabled True --run.steps 400000|runs/fep_atari/pitfall
```

---

**文档版本**: v1.1
**最后更新**: 2026-03-06
**负责人**: User + Claude