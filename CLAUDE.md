# FEP-Unified Dreamer 项目指南

**项目**: 基于自由能原理的统一探索机制
**目标**: ICLR 2027 投稿
**最后更新**: 2026-03-05

---

## 🎯 快速导航

### 📋 核心文档
- **实验计划**: `docs/analysis/experiment_plan.md` - 详细的实验时间表和配置
- **训练分析**: `docs/analysis/fep_training_analysis.md` - 当前训练结果和发现
- **设计文档**: `docs/fep_dreamer_design.md` - FEP 模块设计
- **文档索引**: `docs/README.md` - 所有文档的总览

### 🖼️ 可视化
- `docs/figures/` - 所有生成的图表

### 📊 实验数据
- `runs/` - 所有实验的训练日志和 checkpoint
- Tensorboard: http://hy-127-64:6006

---

## 🔄 每次对话开始时必做

### 1. 检查实验计划
```
阅读: docs/analysis/experiment_plan.md
- 查看当前阶段和里程碑
- 确认下一步任务
- 检查是否有延期风险
```

### 2. 检查运行中的实验
```bash
# 查看所有训练进程
ps aux | grep "python.*main.py" | grep -v grep

# 根据进程找到对应的 logdir，然后查看最新指标
# 示例：如果是 runs/fep_crafter
tail -1 runs/fep_crafter/metrics.jsonl | python3 -m json.tool
```

### 3. 确认当前焦点
```
查看 docs/analysis/experiment_plan.md 中的"检查点和里程碑"
- 当前在哪个阶段？
- 有什么阻塞问题？
- 需要做什么决策？
```

---

## 📍 当前状态（每次对话更新）

**当前阶段**: 阶段 1 - Crafter 完成
**进度**: 25.8% (284k/1.1M 步)
**预计完成**: 2026-03-06
**下一个里程碑**: Crafter FEP 训练完成

**最新发现**: 见 `docs/analysis/fep_training_analysis.md`

---

## ⚡ 通用监控命令

### 查看运行中的实验
```bash
# 列出所有训练进程
ps aux | grep "python.*main.py" | grep -v grep

# 查看 GPU 使用
nvidia-smi

# 查看所有实验目录
ls -lh runs/
```

### 查看特定实验（替换 <experiment_name>）
```bash
# 查看最新指标
tail -1 runs/<experiment_name>/metrics.jsonl | python3 -m json.tool

# 查看最新日志
tail -20 runs/<experiment_name>.log

# 查看训练曲线（需要 tensorboard）
tensorboard --logdir runs/<experiment_name>
```

---

## 🚨 每次对话的检查流程

### 对话开始时（我会主动做）
1. ✅ 检查 `experiment_plan.md` 当前阶段
2. ✅ 查看运行中的实验状态
3. ✅ 确认是否有新的结果需要分析
4. ✅ 识别阻塞问题或需要决策的地方

### 对话过程中（根据讨论）
- 如果有新发现 → 记录到 `fep_training_analysis.md`
- 如果完成任务 → 在 `experiment_plan.md` 中勾选
- 如果计划变更 → 更新 `experiment_plan.md`
- 如果生成图表 → 保存到 `docs/figures/`

### 对话结束时（我会主动做）
1. ✅ 更新本文件的"当前状态"
2. ✅ 更新"最后更新"日期
3. ✅ 确认下次对话的焦点

---

## 📝 快速链接

### 查看待办事项
```
docs/analysis/experiment_plan.md 第 10 节 "下一步行动"
```

### 查看最新分析
```
docs/analysis/fep_training_analysis.md
```

### 查看时间线
```
docs/analysis/experiment_plan.md 第 4 节 "详细时间表"
```

---

## 🎓 文档分工

- **CLAUDE.md** (本文件): 导航、检查清单、当前状态快照
- **experiment_plan.md**: 完整的实验计划、时间表、配置
- **fep_training_analysis.md**: 深入的分析、发现、结论
- **README.md**: 文档索引、项目概览、论文规划

**原则**: 本文件只记录"在哪里找"和"当前在哪"，不重复详细内容！

---

**最后更新**: 2026-03-05
**下次对话**: 我会先检查实验状态和计划进度