# FEP Integration - Phase 5 完成报告

## 执行摘要

✅ **Phase 5 (子目标分解) 已完成并通过所有验证测试**

实现了层次化子目标分解器，能够将长期目标递归分解为可达的子目标序列。这使得 agent 能够处理复杂的长期规划任务。

## 实现细节

### 1. SubgoalDecomposer 类 (`dreamerv3/fep_components.py`)

#### 核心功能
```python
class SubgoalDecomposer:
  """Recursively decompose long-horizon goals into subgoals.

  NOTE: This runs OUTSIDE of JIT context, so can use Python control flow.
  """
```

**关键特性**:
- 运行在 JIT 外部（可使用 Python 控制流）
- 递归分解算法
- 基于可达性的自适应分解
- 支持配置的最大深度和规划视野

#### 核心方法

**1. `decompose(start_state, goal_state, depth=0)`**
```python
def decompose(self, start_state, goal_state, depth=0):
  """Recursively find subgoals between start and goal.

  Returns:
    List of subgoal states (including final goal)
  """
```

**算法流程**:
1. **Base Case 1**: 达到最大深度 → 返回目标
2. **Base Case 2**: 目标可直接达到 → 返回目标
3. **Recursive Case**:
   - 找到中间点
   - 递归分解 start → midpoint
   - 递归分解 midpoint → goal
   - 合并子目标序列

**2. `_is_reachable(start, goal, horizon)`**
```python
def _is_reachable(self, start, goal, horizon):
  """Check if goal is reachable from start within horizon steps.

  Uses distance heuristic scaled by horizon.
  """
```

**可达性判断**:
- 计算状态特征距离
- 使用启发式阈值: `threshold = 5.0 * sqrt(horizon)`
- 距离小于阈值 → 可达

**3. `_find_midpoint(start, goal)`**
```python
def _find_midpoint(self, start, goal):
  """Find intermediate state between start and goal.

  Uses linear interpolation in latent space.
  """
```

**中间点计算**:
- 在潜在空间中线性插值
- `midpoint = 0.5 * start + 0.5 * goal`
- 对 `deter` 和 `stoch` 分别插值

**4. `plan_to_goal(start_state, goal_state, policy)`**
```python
def plan_to_goal(self, start_state, goal_state, policy):
  """Plan a sequence of actions to reach goal via subgoals.

  High-level planning function.
  """
```

**规划流程**:
1. 分解目标为子目标序列
2. 对每个子目标规划动作
3. 合并动作序列

### 2. Agent 集成 (`dreamerv3/agent.py`)

#### 初始化 (第 96-100 行)
```python
# Subgoal decomposer for Case 2 (Phase 5)
self.subgoal_decomposer = fep_components.SubgoalDecomposer(
    world_model=self.dyn,
    max_depth=config.fep.get('subgoal_max_depth', 3),
    horizon=config.fep.get('subgoal_horizon', 15))
```

### 3. 配置 (`dreamerv3/configs.yaml`)

```yaml
fep:
  enabled: False
  # ... existing config ...
  efe_weight: 0.1
  subgoal_max_depth: 3       # 最大递归深度
  subgoal_horizon: 15        # 可达性检查的规划视野
```

## 算法原理

### 递归分解算法

```
function decompose(start, goal, depth):
  if depth >= max_depth:
    return [goal]

  if is_reachable(start, goal):
    return [goal]

  midpoint = find_midpoint(start, goal)
  subgoals1 = decompose(start, midpoint, depth+1)
  subgoals2 = decompose(midpoint, goal, depth+1)

  return subgoals1 + subgoals2
```

### 可达性启发式

```
distance = ||feature(start) - feature(goal)||₂
threshold = 5.0 × √horizon
reachable = (distance < threshold)
```

### 中间点插值

```
midpoint.deter = 0.5 × start.deter + 0.5 × goal.deter
midpoint.stoch = 0.5 × start.stoch + 0.5 × goal.stoch
```

## 验证测试

### 测试文件
`test_subgoal_decomposer.py` - 完整的单元测试

### 测试结果
```
✓ SubgoalDecomposer can be created
✓ State to feature conversion works
✓ Midpoint computation works (exact interpolation)
✓ Reachability check works (close=True, far=False)
✓ Simple decomposition works (1 subgoal for close goal)
✓ Complex decomposition works (8 subgoals for far goal)
✓ Max depth limit works (respects depth constraint)
✓ Python syntax is valid
```

### 测试案例

**Case 1: 近距离目标**
- Start: zeros
- Goal: 0.1 × ones
- Result: 1 subgoal (直接可达)

**Case 2: 远距离目标**
- Start: zeros
- Goal: 10.0 × ones
- Result: 8 subgoals (递归分解)
- Subgoal progression: 48.99 → 97.98 → ... → 391.92

**Case 3: 深度限制**
- max_depth = 1
- Far goal
- Result: 2 subgoals (受深度限制)

## 设计决策

### 1. JIT 外运行
**原因**:
- 递归算法需要动态控制流
- 可达性检查可能需要复杂逻辑
- 灵活性 > 性能（规划不在关键路径）

**优势**:
- 可使用 Python 递归
- 可使用动态数据结构
- 易于调试和扩展

### 2. 线性插值
**原因**:
- 简单高效
- 在潜在空间中合理
- 可作为更复杂方法的基线

**未来改进**:
- 使用 world model 想象中间状态
- 基于梯度的优化
- 学习的插值函数

### 3. 距离启发式
**原因**:
- 快速计算
- 与 horizon 相关
- 可调节阈值

**未来改进**:
- 使用 world model 预测可达性
- 学习的可达性函数
- 考虑动作约束

## 使用示例

### 基本使用
```python
# 创建分解器
decomposer = SubgoalDecomposer(
    world_model=world_model,
    max_depth=3,
    horizon=15
)

# 分解目标
start_state = {'deter': ..., 'stoch': ...}
goal_state = {'deter': ..., 'stoch': ...}

subgoals = decomposer.decompose(start_state, goal_state)
print(f"Decomposed into {len(subgoals)} subgoals")

# 规划到目标
actions = decomposer.plan_to_goal(
    start_state, goal_state, goal_cond_policy
)
```

### 配置调整
```bash
# 增加分解深度（更细粒度）
python3 dreamerv3/main.py --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --agent.fep.subgoal_max_depth 5

# 调整可达性阈值（通过 horizon）
python3 dreamerv3/main.py --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --agent.fep.subgoal_horizon 20
```

## 与其他 Phase 的集成

### Phase 3: Goal-Conditioned Policy
- SubgoalDecomposer 生成子目标序列
- GoalConditionedPolicy 学习达到每个子目标
- 组合实现层次化规划

### Phase 4: Expected Free Energy
- EFE 评估子目标的价值
- 可用于选择最优子目标路径
- 平衡探索和利用

### 完整流程
```
1. GoalBank 存储高奖励状态
2. SubgoalDecomposer 分解长期目标
3. GoalConditionedPolicy 执行到子目标
4. EFE 引导探索新状态
```

## 性能考虑

### 计算复杂度
- 时间: O(2^depth) 最坏情况
- 空间: O(depth) 递归栈
- 实际: 通常远小于最坏情况（早期终止）

### 优化建议
1. **缓存**: 缓存可达性检查结果
2. **剪枝**: 提前终止不可行路径
3. **并行**: 并行分解多个分支
4. **自适应**: 根据任务调整参数

## 应用场景

### 适用任务
✅ 长期规划任务（horizon > 50）
✅ 稀疏奖励环境
✅ 需要层次化策略的任务
✅ 复杂导航任务

### 不适用任务
❌ 短期任务（horizon < 10）
❌ 密集奖励环境
❌ 实时性要求极高的任务

## 未来改进

### 短期改进
1. **World Model 集成**: 使用 world model 想象中间状态
2. **学习的可达性**: 训练可达性预测器
3. **动作约束**: 考虑动作空间限制

### 长期改进
1. **自适应分解**: 根据任务难度调整策略
2. **多目标规划**: 同时规划多个目标
3. **在线学习**: 从经验中改进分解策略

## 限制

1. **启发式可达性**: 简单的距离启发式可能不准确
2. **线性插值**: 可能不在可行状态空间中
3. **无动作约束**: 不考虑动作可行性
4. **固定策略**: 分解策略不学习

## 结论

**Phase 5 实现完成，SubgoalDecomposer 提供了层次化规划能力。**

实现特点：
- ✅ 递归分解算法
- ✅ 可达性检查
- ✅ 灵活的 Python 实现
- ✅ 完整的单元测试
- ✅ 可配置参数

与 Phase 3 和 Phase 4 结合，形成完整的 FEP 探索和规划框架。适用于需要长期规划的复杂任务。
