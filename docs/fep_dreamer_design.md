# FEP 驱动的 Dreamer：统一探索与目标达成

## 1. 动机

DreamerV3 通过 imagination 训练实现了优秀的采样效率，
但其架构分裂为两个独立的优化目标：

- 世界模型：最小化 prediction error（变分自由能）
- Policy：最大化累积外部 reward（标准 RL）

这种分裂导致 **prediction error 的梯度不流入 policy 的学习**。
Policy 永远不知道"世界模型哪里不确定"，也不会"主动去减少不确定性"，
它只学会了"如何在想象中得分"。

从自由能原理（FEP）的视角看，感知和行动应当被**同一个信号**——
prediction error——驱动。本文描述如何在 Dreamer 框架中实现这种统一。

---

## 2. 背景：FEP vs Dreamer

### 2.1 FEP 核心循环

每个时间步 t，agent 做以下事情：

```
1. 预测当前输入：  o_hat_t = predict(s_t)        // 想象
2. 接收实际输入：  o_t = env.observe()             // 现实
3. 计算预测误差：  e_t = o_hat_t - o_t            // 想象与现实的差异

   e_t 同时驱动两个过程：
     d(e)/d(theta)  ->  更新世界模型（让想象贴近现实）
     d(e)/d(a)      ->  更新动作模型（让现实贴近想象）
```

核心：**一个梯度源，两个流向**。不是两个独立的 loss。

### 2.2 Dreamer 已经做了什么（感知/学习）

RSSM 的训练目标等价于最小化变分自由能：

```
F = -E_q[ln p(o_t | s_t)] + KL[q(s_t | o_t, h_t) || p(s_t | h_t)]
    ~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    重建损失（解码器）            KL 散度（posterior vs prior）
```

这一半完全符合 FEP：
- q(s_t | o_t, h_t) = posterior（观察后的状态估计）= RSSM observe
- p(s_t | h_t) = prior（纯靠世界模型预测的状态）= RSSM imagine
- p(o_t | s_t) = 解码器

### 2.3 Dreamer 没有做什么（行动）

在 FEP 中，动作的产生应该是：

```
我预测自己应处于状态 s*（低自由能/偏好状态）
但实际我在状态 s
预测误差 = s* - s
  -> 产生动作 a，从 s 移向 s*
```

但 Dreamer 的 policy 训练是：

```
在 imagination 中展开 rollout -> 收集想象中的 reward -> 对 reward 做 policy gradient
```

这是标准 RL。**prediction error 完全没有参与 policy 的学习**。

### 2.4 结构性对比

```
FEP（Active Inference）：

  一个目标函数 F（自由能）
    d(F)/d(q)      ->  更新信念（感知）
    d(F)/d(theta)  ->  更新模型（学习）
    d(F)/d(a)      ->  产生动作（行动）

  感知、学习、行动，全部是 F 的梯度下降。


DreamerV3：

  目标函数 1：F（自由能）
    d(F)/d(theta_wm)  ->  更新世界模型

  目标函数 2：E[sum r_t]（累积 reward）
    d/d(theta_policy)  ->  更新 policy

  两个目标，两套梯度流。
  唯一的联系是世界模型为 policy 提供想象环境。
```

---

## 3. 设计：FEP 统一的 Dreamer

### 3.1 缺失的梯度路径

需要让 prediction error 流入 policy：

```
当前：
  d(e)/d(theta_wm)       ->  更新世界模型       [已有]
  d(e)/d(theta_policy)   ->  ???                 [缺失]

目标：
  d(e)/d(theta_wm)       ->  更新世界模型       [保留]
  d(e)/d(theta_policy)   ->  更新 policy         [新增]
  d(e)/d(theta_goal)     ->  更新目标想象         [新增]
```

### 3.2 Reward 在 FEP 中的位置

FEP 中没有外部 reward 信号。取而代之的是**先验偏好（prior preference）**：

```
RL 视角：     agent 追逐 reward
FEP 视角：    agent 逃离 surprise
数学等价：    reward = -surprise = ln p(o)
```

"目标想象"模块承担了先验偏好的角色：
agent 想象目标状态，然后行动以减少当前状态与想象目标之间的差异。

### 3.3 统一的 Prediction Error

每个时间步，prediction error e_t 同时具有三重含义：

```
e_t 大，同时意味着：
  1. 世界模型需要学习      （认知不足）
  2. 动作模型需要调整      （当前行为导致了意外）
  3. 目标想象需要修正      （之前的想象不现实）
```

一个信号驱动一切。

### 3.4 Policy 目标的重新表述

在 imagination rollout（imag_loss）中，修改 policy 的优化目标：

```
# 原版 DreamerV3：policy 目标是纯 reward
policy_target = reward

# FEP 统一框架：policy 目标包含 prediction error
policy_target = (
    reward                                  # 利用：追求回报
    - alpha * expected_prediction_error     # 探索：去误差大的地方
    + beta * goal_proximity                 # 目标：接近想象的目标
)
```

这三项都可以从 prediction error 推导出来：

```
reward             ~=  -surprise  ~=  -preferred state 处的 prediction error
exploration signal ~=  expected prediction error reduction（信息增益）
goal signal        ~=  -当前状态与目标之间的 prediction error
```

### 3.5 目标想象的 Bootstrapping

目标想象随着世界模型一起进化：

```
初始阶段：
  世界模型很差，想象出的"目标状态"模糊/不正确。
  但它提供了一个粗糙的方向。

探索阶段：
  Agent 向模糊方向移动 -> 遇到新状态 -> 世界模型更新
  -> 重新想象目标状态 -> 比之前更准确
  -> 向更准确的方向移动 -> 又遇到新状态
  ...

收敛阶段：
  世界模型足够好，想象出的目标状态 ~= 真实目标状态。
  Agent 可以在 imagination 中规划完整路径。
```

这形成了一个自我改进的循环：

```
模糊目标 -> 粗糙探索 -> 更好的世界模型 -> 更清晰的目标 -> 更精确的探索 -> ...
```

---

## 4. 架构

### 4.1 完整循环

```
                    prediction error
                    （想象 vs 现实）
                   /         |          \
                  /          |           \
          更新世界模型    更新目标想象    更新动作模型
          （想象更准）   （目标更实际）  （行为更有效）
               \             |            /
                \            |           /
                 -> 更准的世界模型 + 更现实的目标
                             |
                       imagination rollout
                       （在更准的世界模型中用更现实的目标规划）
                             |
                       产生动作 -> 环境交互 -> 新的 prediction error
                             |
                          循环回到顶部
```

### 4.2 与之前方案的核心区别

```
之前的方案（松散耦合）：
  世界模型训练  <-  prediction error
  Policy 训练   <-  reward（独立的信号源）
  目标想象      <-  独立模块
  三者松散连接。

修正后的方案（统一驱动）：
  世界模型训练  <-  prediction error
  Policy 训练   <-  prediction error（同一个信号）
  目标想象      <-  prediction error（同一个信号）
  三者被 prediction error 统一驱动。
```

### 4.3 对应到 Dreamer 代码结构

```
agent.py:
  loss()          ->  将 prediction error 项加入 imagination reward
  imag_loss()     ->  修改 policy 目标，包含 EFE 成分
  _compute_curiosity_action()  ->  替换为基于 EFE 的动作选择

rssm.py:
  RSSM.loss()     ->  暴露 KL[posterior || prior] 作为信息增益信号
  RSSM.imagine()  ->  沿想象轨迹计算 expected prediction error

新模块 (goal_imagination.py):
  GoalImaginator  ->  从 task description + 当前世界模型生成 z_goal
                      基于 prediction error 反馈更新 z_goal
```

---

## 5. 具体实现方案

### 5.1 Prediction Error 作为内在信号

在 RSSM 中，prediction error 在训练时已经被计算：

```
e_t = KL[q(z_t | o_t, h_t) || p(z_t | h_t)]
```

这个 KL 散度就是信息增益——agent 从观察 o_t 中学到了多少。
目前它只用来训练世界模型。

第一步：暴露这个信号，将其注入 imagination rollout 作为额外的 reward 项。

### 5.2 Imagination 中的 Expected Prediction Error

在 imagination 中无法计算真实的 posterior（没有真实观察），
但可以近似 expected prediction error：

```
方案 A：Ensemble disagreement（K 个模型，计算方差）
  - 优点：干净的 epistemic uncertainty 估计
  - 缺点：K 倍计算开销

方案 B：Prior 熵 H(p(z' | h'))
  - 优点：零额外参数，RSSM 中已有
  - 缺点：混合了 epistemic 和 aleatoric uncertainty

方案 C：学习一个 prediction error 预测器
  - 训练一个 head，在只有 prior（无观察）的情况下
    预测 KL[posterior || prior] 会有多大
  - 优点：直接预测信息增益，单模型
  - 缺点：需要训练额外的 head
```

方案 C 最符合 FEP 精神：它问的是"如果我去了这个状态，
我能学到多少？"而不需要实际的观察。

### 5.3 目标想象模块

```python
# Conceptual structure
class GoalImaginator:
    """Generate and refine goal states in latent space.

    Goal states are generated by the world model and refined as the
    world model improves. This implements the bootstrapping loop:
    vague goal -> exploration -> better model -> clearer goal.
    """

    def imagine_goal(self, task_embedding, world_model_state):
        """Generate goal state z_goal in latent space.

        Early in training: z_goal is vague (high entropy)
        Late in training: z_goal is specific (low entropy)
        """
        ...

    def update(self, prediction_error, achieved_states):
        """Refine goal imagination based on prediction error.

        When prediction error is high near goal-relevant states,
        the goal imagination adjusts to be more realistic.
        """
        ...
```

### 5.4 修改后的 Imagination Loss

```python
def imag_loss_fep(imgfeat, ...):
    # 标准 reward（利用）
    reward_ext = reward_head(imgfeat).pred()

    # 沿想象轨迹的信息增益（探索）
    # 用 learned prediction error predictor 近似
    info_gain = info_gain_head(imgfeat).pred()

    # latent space 中的目标接近度（目标导向行为）
    goal_dist = -||imgfeat - z_goal||^2

    # 统一目标（全部源自 prediction error）
    policy_target = reward_ext + alpha * info_gain + beta * goal_dist

    # alpha 和 beta 可以动态调节：
    # - 训练早期：alpha 大（多探索），beta 小（目标模糊）
    # - 训练后期：alpha 小（多利用），beta 大（目标清晰）
```

---

## 6. 与已有工作的关系

| 方法 | 探索信号 | 是否修改训练？ | 是否符合 FEP？ |
|------|---------|---------------|---------------|
| 当前项目（curiosity trigger） | Prior 熵 | 否（仅 action selection） | 部分 |
| Plan2Explore | Ensemble disagreement | 是（intrinsic reward） | 部分 |
| DreamerV3-XP | Reward ensemble variance | 是（intrinsic reward） | 部分 |
| Enter the Void | 转移不确定性 | 是（对抗性 WM 目标） | 部分 |
| LEXA | Goal-conditioned + latent disagreement | 是 | 部分 |
| **本方案** | **Prediction error（统一）** | **是（WM + policy + goal）** | **是** |

核心差异：已有方法将探索作为 RL 之上的"补丁"。
本方案让 prediction error 成为所有学习和行动的**唯一驱动力**，
这是 FEP 的核心主张。

---

## 7. 待解决问题

### 7.1 Epistemic vs Aleatoric Uncertainty

单模型 prior 熵混合了 epistemic（可消除的）和 aleatoric（不可消除的）不确定性。
"噪声电视问题"：agent 盯着随机噪声看，因为 prediction error 始终很高。

可能的解决方案：
- Ensemble disagreement（过滤 aleatoric，但 K 倍计算开销）
- KL[posterior || prior]（天然只捕捉 epistemic，但需要真实观察）
- 学习的 info gain 预测器（上述方案 C，在无观察时近似 KL）

### 7.2 目标想象的冷启动

世界模型还没见过任何与 reward 相关的状态时，如何生成有意义的 z_goal？

可能的解决方案：
- 用 task description embedding 作为初始先验
- 以最大熵 z_goal 开始（广泛探索）
- 用 hindsight relabeling（任何到达过的状态都可以是"目标"）

### 7.3 alpha 和 beta 的平衡

静态系数可能无法适应不同训练阶段。需要自适应调度：
- 基于熵：随着世界模型 prediction error 下降而降低 alpha
- 基于进度：随着目标想象置信度提升而增加 beta
- 基于 EMA：复用现有 CuriosityTrigger 的机制来做调度

---

## 8. 总结

```
核心论点：
  Dreamer 将感知（prediction error）和行动（reward）分裂为
  两个独立的目标。FEP 要求它们被统一。

提出的方案：
  让 prediction error 同时驱动三件事：
    1. 世界模型学习     （想象变得更准确）
    2. Policy 学习      （行动变得更有效）
    3. 目标想象         （目标变得更现实）

  这创建了一个自我改进的循环：
    模糊目标 -> 粗糙探索 -> 更好的世界模型
    -> 更清晰的目标 -> 精确探索 -> 达成目标

与已有工作的关键区别：
  不是 "RL + curiosity 补丁"，
  而是 "prediction error 驱动一切"。
  目标想象从世界模型自身中 bootstrap 出来，
  随着世界模型的改进而自动变得更现实。
```
