# FEP Integration for DreamerV3

Free Energy Principle (FEP) 集成到 DreamerV3，提供理论驱动的探索和规划框架。

## 🎉 项目状态

**✅ 全部完成** - 所有 5 个阶段已实现、测试并验证

## 快速开始

### 启用 FEP
```bash
python3 dreamerv3/main.py --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --run.steps 100000
```

### 运行测试
```bash
# 完整验证
python3 test_final_verification.py

# 单独测试
python3 test_fep_jit.py              # GoalBank
python3 test_goal_policy.py          # Goal-Conditioned Policy
python3 test_efe.py                  # Expected Free Energy
python3 test_subgoal_decomposer.py   # Subgoal Decomposition
```

## 核心组件

### 1. GoalBank (Phase 2)
存储和采样高奖励状态
- 固定大小存储 (capacity=100)
- JIT 兼容
- 用于提供探索目标

### 2. Goal-Conditioned Policy (Phase 3)
学习如何达到目标状态
- π(a|s, g) 目标条件策略
- Behavior cloning 训练
- 与主策略相同的 MLP 架构

### 3. Expected Free Energy (Phase 4)
FEP 驱动的探索奖励
- EFE = Epistemic Value + Pragmatic Value
- Epistemic: 信息增益（探索）
- Pragmatic: 预期奖励（利用）

### 4. Subgoal Decomposer (Phase 5)
层次化长期规划
- 递归分解长期目标
- 可达性检查
- 运行在 JIT 外（灵活）

## 配置参数

```yaml
fep:
  enabled: False              # 全局开关
  alpha_max: 0.1              # Epistemic 权重
  beta_max: 0.05              # Pragmatic 权重
  efe_weight: 0.1             # EFE bonus 权重
  subgoal_max_depth: 3        # 子目标最大深度
  subgoal_horizon: 15         # 可达性检查视野

loss_scales:
  goal_policy: 0.5            # Goal policy loss 权重
```

## 监控指标

训练时可监控的指标：
- `fep/alpha` - Epistemic 权重
- `fep/beta` - Pragmatic 权重
- `fep/img_info_gain` - 信息增益
- `fep/img_goal_prox` - 目标接近度
- `fep/efe_mean` - EFE 平均值
- `fep/efe_std` - EFE 标准差
- `loss/goal_policy` - Goal policy 损失

## 文档

详细文档请参阅：
- `PROJECT_COMPLETION_REPORT.md` - 项目完成报告
- `FEP_INTEGRATION_SUMMARY.md` - 总体摘要
- `PHASE3_COMPLETION_REPORT.md` - Phase 3 详细报告
- `PHASE4_COMPLETION_REPORT.md` - Phase 4 详细报告
- `PHASE5_COMPLETION_REPORT.md` - Phase 5 详细报告

## 适用场景

### ✅ 推荐使用
- 稀疏奖励环境
- 长期规划任务
- 需要探索的任务
- 复杂导航任务

### ❌ 不推荐使用
- 密集奖励环境
- 短期任务
- 实时性要求极高的任务

## 架构

```
DreamerV3 Agent
├── World Model (RSSM)
├── Policy (Actor)
├── Critic (Value)
└── FEP Components
    ├── GoalBank
    ├── Goal-Conditioned Policy
    ├── Expected Free Energy
    └── Subgoal Decomposer
```

## 技术特性

- ✅ JAX JIT 兼容（Phase 2-4）
- ✅ 模块化设计
- ✅ 向后兼容
- ✅ 完整测试覆盖
- ✅ 详细文档

## 性能

| 组件 | 计算开销 | 内存开销 |
|------|---------|---------|
| GoalBank | 低 | 固定 |
| Goal-Cond Policy | 中 | 小 |
| EFE | 中 | 小 |
| Subgoal Decomposer | 低 | 小 |

## 示例

### 调整探索参数
```bash
python3 dreamerv3/main.py --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --agent.fep.alpha_max 0.2 \
  --agent.fep.beta_max 0.1 \
  --agent.fep.efe_weight 0.15
```

### 调整规划参数
```bash
python3 dreamerv3/main.py --configs dmc_vision \
  --task dmc_walker_walk \
  --agent.fep.enabled True \
  --agent.fep.subgoal_max_depth 5 \
  --agent.fep.subgoal_horizon 20
```

## 理论基础

基于 Free Energy Principle (FEP)：
- Agent 通过最小化 free energy 来适应环境
- Epistemic Value: 减少不确定性（探索）
- Pragmatic Value: 最大化奖励（利用）

## 贡献

本项目由 Claude (Anthropic) 协助实现。

## 许可证

遵循 DreamerV3 的原始许可证。

## 致谢

- DreamerV3 团队
- Free Energy Principle 理论框架
- JAX 和 Ninjax 库

---

**状态**: ✅ 完成并可用
**版本**: 1.0.0
**最后更新**: 2025
