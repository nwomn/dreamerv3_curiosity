# 好奇心机制模块化重构方案

## 总体目标
将当前硬编码的好奇心机制重构为可配置的模块化系统，方便在不同Benchmark（Atari、DMLab、Crafter、DMC等）上快速启用/禁用和调参。

## 设计原则
1. **最小侵入性**: 尽量减少对Agent和RSSM核心代码的改动
2. **配置驱动**: 通过配置文件控制不同Benchmark的好奇心设置
3. **向后兼容**: 保持当前已修复的动态阈值机制功能
4. **易扩展**: 为未来添加新策略（如RND、Count-based）预留接口

## 重构分为两个阶段

### 阶段1: 配置系统重构（快速实用）
**目标**: 让每个Benchmark能独立配置好奇心参数，无需修改代码

**修改内容**:

1. **重构 configs.yaml 结构**
   - 将好奇心配置嵌套为独立模块
   - 为每个Benchmark添加推荐的好奇心配置预设

2. **添加 Benchmark 预设配置**
   - `atari`: 启用好奇心，适中参数
   - `dmlab`: 启用好奇心，高探索倾向
   - `dmc_vision`: 禁用好奇心（密集奖励）
   - `crafter/minecraft`: 启用好奇心，长期探索设置
   - `atari100k`: 启用但采样数减少（数据效率）

3. **创建配置模板文件**
   - `configs/curiosity_presets.yaml`: 为不同类型任务提供预设

4. **最小化 Agent 代码修改**
   - 将好奇心逻辑提取为 `_compute_curiosity_action()` 私有方法
   - 保持主流程清晰

**配置示例**:
```yaml
# configs.yaml 新增结构
curiosity:
  enabled: True
  samples: 10
  alpha: 0.001
  std_scale: 1.0

# Benchmark预设
atari:
  curiosity.enabled: True
  curiosity.std_scale: 1.5  # Atari适中探索

dmlab:
  curiosity.enabled: True
  curiosity.std_scale: 2.0  # 更激进探索
  curiosity.samples: 15

dmc_vision:
  curiosity.enabled: False  # 密集奖励不需要

crafter:
  curiosity.enabled: True
  curiosity.std_scale: 1.2
  curiosity.alpha: 0.0005   # 更慢适应
```

### 阶段2: 模块化架构重构（长期维护）
**目标**: 创建可插拔的好奇心策略框架

**新增文件结构**:
```
dreamerv3/
├── curiosity/
│   ├── __init__.py
│   ├── base.py              # 抽象基类
│   ├── entropy_based.py     # 当前实现（动态阈值）
│   └── disabled.py          # 空实现（无好奇心）
```

**修改内容**:

1. **创建 curiosity/base.py**
   - 定义 `CuriosityStrategy` 抽象基类
   - 标准化接口: `initial_state()`, `select_action()`, `get_metrics()`

2. **创建 curiosity/entropy_based.py**
   - 将当前 CuriosityTrigger 和相关逻辑迁移
   - 封装 RSSM 不确定性计算
   - 保持无状态设计

3. **创建 curiosity/disabled.py**
   - 空操作实现，无性能开销
   - 用于禁用好奇心的Benchmark

4. **修改 agent.py**
   - 添加 `_make_curiosity()` 工厂方法
   - 简化 `policy()` 方法，调用策略对象
   - 清理注释掉的旧代码

5. **更新配置系统**
   ```yaml
   curiosity:
     strategy: entropy  # 策略选择器
     entropy:           # 策略专属配置
       samples: 10
       alpha: 0.001
       std_scale: 1.0
   ```

## 具体实施步骤

### 步骤1: 配置系统重构
1. 重构 `configs.yaml` - 嵌套好奇心配置
2. 为10个主要Benchmark添加预设配置
3. 创建 `configs/benchmark_curiosity.yaml` 预设模板
4. 更新 README 说明配置方法

### 步骤2: Agent代码清理
1. 提取 `_compute_curiosity_action()` 方法
2. 删除注释掉的 Method 1 代码（Lines 143-165）
3. 删除未使用的 `curiosity_sample()` 和 `sample_uniform_actions()`
4. 简化 `policy()` 方法主流程

### 步骤3: 创建好奇心模块
1. 创建 `dreamerv3/curiosity/` 目录
2. 实现 `base.py` 抽象接口
3. 迁移当前实现到 `entropy_based.py`
4. 实现 `disabled.py` 空策略
5. 创建 `__init__.py` 导出接口

### 步骤4: 集成到Agent
1. 实现 `Agent._make_curiosity()` 工厂
2. 替换 `policy()` 中的内联逻辑
3. 更新 `init_policy()` 使用策略对象
4. 确保向后兼容性

### 步骤5: 测试和验证
1. 在 Atari (Pong) 上测试启用好奇心
2. 在 DMC 上测试禁用好奇心
3. 验证配置覆盖机制
4. 运行原有的 bug 修复测试

## 文件修改清单

### 需要修改的文件
- `dreamerv3/configs.yaml` - 重构配置结构
- `dreamerv3/agent.py` - 提取好奇心逻辑、添加工厂方法
- `dreamerv3/rssm.py` - (可选) 添加文档说明不确定性方法

### 需要创建的文件
- `dreamerv3/curiosity/__init__.py`
- `dreamerv3/curiosity/base.py`
- `dreamerv3/curiosity/entropy_based.py`
- `dreamerv3/curiosity/disabled.py`
- `configs/benchmark_curiosity.yaml` (可选预设模板)
- `CURIOSITY_GUIDE.md` (使用文档)

### 需要删除的代码
- `agent.py` Lines 143-165 (注释掉的 Method 1)
- `agent.py` Lines 206-211 (`curiosity_sample()` 未使用)
- `agent.py` 中的 `sample_uniform_actions()` (如果确认未使用)

## 预期效果

### 使用示例

**场景1: 在Atari上启用默认好奇心**
```bash
python dreamerv3/main.py --configs atari --task atari_montezuma_revenge
# 自动使用atari预设的好奇心配置
```

**场景2: 禁用好奇心测试基线**
```bash
python dreamerv3/main.py --configs atari --curiosity.enabled False
```

**场景3: 调整好奇心参数**
```bash
python dreamerv3/main.py --configs dmlab --curiosity.std_scale 2.5 --curiosity.samples 20
```

**场景4: 对比实验**
```bash
# 实验组
python dreamerv3/main.py --configs crafter --logdir logs/crafter_curiosity

# 对照组
python dreamerv3/main.py --configs crafter --curiosity.enabled False --logdir logs/crafter_baseline
```

### 收益
1. **零代码改动切换Benchmark**: 只需修改 `--configs` 参数
2. **清晰的配置文件**: 一目了然每个Benchmark的设置
3. **灵活的参数调优**: 命令行快速override
4. **便于消融实验**: 轻松对比启用/禁用好奇心
5. **代码更清晰**: 主流程与好奇心逻辑分离

## 时间估计
- **阶段1** (配置重构): 2-3小时
- **阶段2** (模块化重构): 4-6小时
- **测试验证**: 2-3小时
- **总计**: 1-2个工作日

## 建议
推荐**先执行阶段1**，快速获得配置灵活性，满足立即的Benchmark测试需求。阶段2可以后续进行，适合长期维护和添加新策略时实施。

---

## 附录：可用的Benchmark列表

基于对项目的分析，以下是可以测试好奇心机制的主要Benchmark：

### 1. Atari (57 games)
- **适合好奇心**: ✅ 是（尤其是稀疏奖励游戏如Montezuma's Revenge）
- **推荐配置**: enabled=True, std_scale=1.5
- **示例任务**: `atari_pong`, `atari_montezuma_revenge`, `atari_breakout`

### 2. DeepMind Lab (30 tasks)
- **适合好奇心**: ✅✅ 非常适合（3D导航和探索）
- **推荐配置**: enabled=True, std_scale=2.0, samples=15
- **示例任务**: `dmlab_explore_goal_locations_small`

### 3. Crafter
- **适合好奇心**: ✅✅ 非常适合（开放世界，长期稀疏奖励）
- **推荐配置**: enabled=True, std_scale=1.2, alpha=0.0005
- **示例任务**: `crafter_reward`

### 4. Minecraft
- **适合好奇心**: ✅✅ 非常适合（探索驱动）
- **推荐配置**: enabled=True, std_scale=1.5
- **示例任务**: `minecraft_diamond`

### 5. ProcGen (16 games)
- **适合好奇心**: ✅ 适合（程序生成，需要泛化）
- **推荐配置**: enabled=True, std_scale=1.3
- **示例任务**: `procgen_coinrun`, `procgen_maze`

### 6. DeepMind Control (DMC)
- **适合好奇心**: ❌ 不太需要（密集奖励，连续控制）
- **推荐配置**: enabled=False
- **示例任务**: `dmc_walker_walk`, `dmc_cheetah_run`

### 7. Atari100k (数据效率测试)
- **适合好奇心**: ✅ 适合（数据受限场景）
- **推荐配置**: enabled=True, samples=5, std_scale=1.2
- **示例任务**: 同Atari，但限制100k步

### 8. BSuite (行为套件)
- **适合好奇心**: ⚠️ 视任务而定
- **推荐配置**: enabled=False (默认)
- **示例任务**: `bsuite_mnist/0`

## 优先测试建议

**高优先级（最能体现好奇心价值）**:
1. Atari: Montezuma's Revenge, Venture (稀疏奖励)
2. DMLab: 探索任务
3. Crafter: 完整任务

**中优先级（对比实验）**:
4. Atari: Pong, Breakout (密集奖励对照)
5. ProcGen: 泛化能力测试

**基线对照**:
6. DMC: 验证禁用好奇心不影响性能
