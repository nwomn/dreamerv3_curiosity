# 三大流派架构对比

## 1. 整体信息流

### 流派一：标准 RL + 世界模型（Dreamer）

```
环境 ──o_t──→ [编码器] ──→ [RSSM世界模型] ──→ [解码器] ──→ ô_t
                                  │
                            prediction error ──→ 更新世界模型（唯一用途）
                                  │
                            imagination rollout
                                  │
                            [reward head] ──→ r_hat
                                  │
                            policy gradient on r_hat ──→ 更新 policy
                                  │
                            [value head] ──→ V(s)

两条独立的优化链：
  链 A：prediction error → 世界模型参数
  链 B：imagined reward → policy 参数
```

### 流派二：Deep Active Inference（R-AIF）

```
环境 ──o_t──→ [编码器] ──→ [RSSM世界模型] ──→ [解码器] ──→ ô_t
                                  │
                            prediction error ──┬──→ 更新世界模型
                                  │            │
                            imagination rollout │
                                  │            │
                                 EFE ←─────────┘
                             (= epistemic + pragmatic)
                                  │
                            EFE gradient ──→ 更新 policy
                                  │
                      [先验偏好模型] ──→ "我应该在什么状态"

一条主链（但 pragmatic 部分仍需外部 reward/偏好）：
  prediction error → EFE → 世界模型 + policy
```

### 流派三：目标想象 + 探索（LEXA / MUN）

```
环境 ──o_t──→ [编码器] ──→ [世界模型] ──→ [解码器] ──→ ô_t
                                  │
                            prediction error ──→ 更新世界模型
                                  │
                       ┌──── imagination ────┐
                       │                     │
                 [Explorer]             [Achiever]
                 在想象中找新目标       在想象中练习达成目标
                       │                     │
                 intrinsic reward       extrinsic reward
                 (disagreement等)       (goal distance等)
                       │                     │
                 更新 explorer policy   更新 achiever policy

三条独立的优化链：
  链 A：prediction error → 世界模型
  链 B：intrinsic reward → explorer
  链 C：extrinsic reward / goal reward → achiever
```

---

## 2. 模块级对比

```
┌─────────────┬───────────────────┬───────────────────┬───────────────────┐
│             │ Dreamer           │ Deep AIF          │ 目标想象+探索      │
│             │ (DreamerV3)       │ (R-AIF)           │ (LEXA/MUN)        │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 世界模型     │ RSSM              │ RSSM              │ RSSM              │
│             │ (posterior+prior)  │ (posterior+prior)  │ (posterior+prior)  │
│             │                   │                   │ + ensemble(可选)   │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 世界模型目标 │ VFE               │ VFE               │ VFE               │
│             │ = recon + KL      │ = recon + KL      │ = recon + KL      │
│             │                   │ (或 contrastive)  │                   │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Policy 数量 │ 1 个              │ 1 个              │ 2 个              │
│             │ (actor)           │ (actor)           │ (explorer+achiever)│
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Policy 目标 │ max E[sum r_t]    │ min EFE           │ explorer: max 新奇│
│             │ (纯 reward)       │ (= reward +       │ achiever: min 到  │
│             │                   │   信息增益)        │   目标的距离       │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 探索机制     │ 熵正则化          │ EFE 中的          │ 独立的 explorer   │
│             │ (actent, 极弱)    │ epistemic term    │ + disagreement    │
│             │                   │ (理论上优雅)       │ (工程上有效)       │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 目标表征     │ 无                │ prior preference  │ latent goal z_g   │
│             │ (只追 reward)     │ (偏好状态分布)     │ (随机/学习/外部)   │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ prediction  │ 只训练世界模型    │ 训练世界模型       │ 只训练世界模型     │
│ error 用途  │                   │ + 驱动 policy      │ (explorer 用      │
│             │                   │ (通过 EFE)         │  disagreement,    │
│             │                   │                   │  不是 pred error)  │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 额外模块     │ reward head       │ prior preference  │ goal sampler      │
│             │ value head        │ model (CRSPP等)   │ goal-conditioned  │
│             │                   │ (学习偏好状态)     │ value/policy      │
├─────────────┼───────────────────┼───────────────────┼───────────────────┤
│ reward 角色 │ 唯一驱动力        │ EFE 的 pragmatic  │ achiever 的       │
│             │                   │ term 之一          │ 驱动力之一         │
└─────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

## 3. 各流派的核心优势和结构性缺陷

### 流派一：Dreamer（标准 RL + 世界模型）

**优势：**
- 结构简洁 —— 模块少、接口清晰、调试容易
- 工程成熟 —— 150+ 任务单一配置，Nature 发表
- imagination 高效 —— 纯在 latent space 中 rollout，不需解码

**结构性缺陷：**

```
世界模型           policy
   │                  │
   │   ██████████     │
   │   █ 信息墙 █     │
   │   ██████████     │
   ↓                  ↓
min VFE           max reward

世界模型知道"我哪里不确定"，
但这个信息被一堵墙挡住了，policy 看不到。
policy 只看到 reward head 的输出。
```

结果：
- sparse reward 环境下探索能力极弱
- 世界模型在已知区域越来越准，在未知区域越来越差
- 形成恶性循环：不探索 → 数据偏 → 模型偏 → 更不探索

### 流派二：Deep AIF（Active Inference）

**优势：**
- 统一目标 —— EFE 一个函数同时包含探索和利用
- 理论优雅 —— prediction error 自然流入 policy
- 无需 reward shaping —— 没有外部 reward 也能通过信息增益驱动行为

**结构性缺陷：**

```
EFE = epistemic + pragmatic
       │              │
   信息增益         实用价值
   (天然有)        (从哪来？)
                       │
                  prior preference
                  (需要人为指定或学习)
```

- 问题 1：pragmatic term 的来源。FEP 说 reward = -surprise = ln p(o)，
  但 p(o) 这个 prior preference 本质上还是要人为设计，和 reward shaping 换了一种说法。
- 问题 2：EFE 的计算开销。完整的 EFE 需要对每个候选动作序列评估未来自由能，
  传统做法是枚举所有 policy（不可扩展），用 amortized policy 近似则质量不可控。
- 问题 3：目标缺乏自主生成。prior preference 是固定的或从 reward 信号学来的，
  没有"目标随着世界模型改进而变得更清晰"的动态过程。

### 流派三：目标想象 + 探索（LEXA / MUN）

**优势：**
- 双策略分工 —— explorer 负责发现，achiever 负责执行
- 目标多样性 —— 可以生成大量不同目标进行练习
- zero-shot 迁移 —— 预训练后直接给新任务的 goal image 就能执行

**结构性缺陷：**

```
[Explorer]           [Achiever]
   │                     │
intrinsic reward     goal distance
   │                     │
独立训练              独立训练
   │                     │
   └───── 弱耦合 ────────┘
```

- 问题 1：目标的质量。LEXA 中目标是从 latent prior 随机采样的，
  大量目标是无意义的，浪费 explorer 的训练资源。
- 问题 2：explorer 和 achiever 的脱节。explorer 发现了新状态，
  但 achiever 不一定能到达；两者之间缺乏紧密信息反馈。
- 问题 3：没有统一的驱动力。explorer 用 disagreement，achiever 用 goal distance，
  世界模型用 prediction error，三个模块三套目标，各自为政。
- 问题 4：不知道"什么不知道"。explorer 的 disagreement 只衡量"模型们意见不一致的地方"，
  不等于"对达成最终目标有价值的未知区域"。

---

## 4. 根本哲学差异

```
Dreamer 的世界观：
  "我有一个模拟器（世界模型），我在里面练习怎么得高分。"
  世界模型是工具，reward 是目标。
  类比：考生在模拟题上刷分，不关心哪些知识点不会。

Deep AIF 的世界观：
  "我有一个关于世界的信念，我行动使现实符合信念。"
  世界模型是信念本身，surprise 最小化是目标。
  类比：科学家根据理论预测实验结果，实验不符就修正理论或改变条件。

目标想象的世界观：
  "我想去一个地方，但我先要知道有哪些地方可去。"
  世界模型是地图，发现新目标和到达目标是两个并行任务。
  类比：探险家先画地图（explorer），再规划路线到达目的地（achiever）。
```

---

## 5. 本方案在对比中的位置

```
                 Dreamer      Deep AIF      目标想象       本方案
                 ───────      ────────      ────────       ────────
世界模型训练      VFE ✓        VFE ✓         VFE ✓          VFE ✓

prediction error
流入 policy      ✗            ✓(EFE)        ✗              ✓(EFE)

目标自主生成      ✗            ✗(需外部偏好)  部分(随机/HER)  ✓(bootstrap)

目标随 WM 改进    ✗            ✗             ✗              ✓

统一驱动力        ✗(两套目标)  部分(EFE统一   ✗(三套目标)    ✓(prediction
                              但偏好仍外部)                  error 统一)

exploration 机制  熵正则(弱)   EFE epistemic  独立explorer   EFE epistemic
                              term(强)       (强但脱节)     + 目标引导

模块数量          少(简洁)     中             多(复杂)       中
```

本方案的融合策略：

```
取 Dreamer 的：  RSSM 世界模型 + imagination 训练（工程基础）
取 Deep AIF 的： EFE 统一目标 + prediction error 流入 policy（理论框架）
取 目标想象的：  目标在 latent space 中生成并自我改进（目标机制）
去掉各自的：    Dreamer 的信息墙、AIF 的外部偏好依赖、目标想象的模块脱节
```

三个流派各自的最大弱点，以及本方案如何回应：

```
Dreamer 的病：    policy 看不到世界模型的不确定性
  → 本方案回应：  prediction error 通过 EFE 直接流入 policy

Deep AIF 的病：   prior preference 从哪来？（需要外部指定或学习）
  → 本方案回应：  目标从世界模型自身 bootstrap，随学习自动改进

目标想象的病：    explorer/achiever/世界模型 三者脱节
  → 本方案回应：  prediction error 统一驱动所有模块
```

---

## 6. 相关文献

### 流派一：标准 RL + 世界模型
- DreamerV3: Hafner et al., "Mastering Diverse Domains through World Models", Nature 2025
  https://danijar.com/project/dreamerv3/
- Dreamer 4: Hafner et al., "Training Agents Inside of Scalable World Models", 2025
  https://arxiv.org/abs/2509.24527
- DIAMOND: "Diffusion for World Modeling: Visual Details Matter in Atari", NeurIPS 2024
  https://diamond-wm.github.io/

### 流派二：Deep Active Inference
- R-AIF: "Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models", 2024
  https://arxiv.org/abs/2409.14216
- Contrastive Active Inference: Mazzaglia et al., NeurIPS 2021
  https://github.com/mazpie/contrastive-aif
- Deep AIF + MTRSSM: "Deep Active Inference with Diffusion Policy and MTRSSM", 2025
  https://arxiv.org/abs/2510.23258
- Deep AIF Delayed: "Deep Active Inference Agents for Delayed and Long-Horizon Environments", 2025
  https://arxiv.org/abs/2505.19867
- Tschantz et al., "Learning action-oriented models through active inference", PLOS Comp Bio 2020
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007805
- VERSES AXIOM: Karl Friston et al., 2025
  https://www.verses.ai/

### 流派三：目标想象 + 探索
- LEXA: Mendonca et al., "Discovering and Achieving Goals via World Models", NeurIPS 2022
  https://orybkin.github.io/lexa/
- MUN: "Learning World Models for Unconstrained Goal Navigation", NeurIPS 2024
- Act2Goal: "From World Model To General Goal-conditioned Policy", 2025
  https://arxiv.org/html/2512.23541v1
- Plan2Explore: Sekar et al., "Planning to Explore via Self-Supervised World Models", ICML 2020
  https://github.com/ramanans1/plan2explore
- RIG: Nair et al., "Visual Reinforcement Learning with Imagined Goals", 2018
  https://arxiv.org/abs/1807.04742
