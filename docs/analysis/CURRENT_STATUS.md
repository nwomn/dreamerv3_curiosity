# 当前实验状态

**更新时间**: 2026-03-06 18:30

## 策略调整

**原策略**: 先跑FEP，再跑Baseline
**新策略**: 先跑Baseline建立基准，再决定FEP投入

**调整原因**:
- FEP有效性尚未充分验证
- 先建立baseline基准，便于后续对比
- 避免在未验证方法上过度投入资源

## 当前运行实验

### 1. Crafter Baseline
- **服务器**: 原服务器
- **GPU**: GPU 1
- **状态**: 🔄 运行中
- **预计完成**: 约1.7天

### 2. Montezuma Baseline
- **服务器**: 新服务器
- **GPU**: GPU 3
- **状态**: 🔄 运行中（刚启动）
- **配置**: CUDA_VISIBLE_DEVICES=3
- **进程PID**: 3018419
- **预计完成**: 约0.62天（15小时）
- **日志**: runs/baseline_atari_montezuma.log

## 已完成实验

### 1. Crafter FEP ✅
- **完成时间**: 2026-03-06
- **训练步数**: 1,099,811 / 1,100,000
- **最终分数**: 9.05分（最后20个episode平均）
- **评估结果**: 8.73分（421个episodes平均）
- **Achievement覆盖率**: 68.2% (15/22)

## 下一步计划

1. **等待Baseline完成**
   - Crafter Baseline（约1.7天）
   - Montezuma Baseline（约0.62天）

2. **评估Baseline结果**
   - 对比Crafter FEP vs Baseline
   - 评估FEP的改进幅度

3. **决定后续方向**
   - 如果FEP有明显优势 → 继续Atari-5 Baseline
   - 如果FEP优势不明显 → 调整FEP配置或方法
   - 如果FEP无效 → 重新评估研究方向

## 实验队列（待定）

**Atari-5 Baseline**:
- Pitfall
- Venture
- PrivateEye
- Solaris

**DMC-5 Baseline**:
- walker_walk
- cheetah_run
- hopper_hop
- quadruped_walk
- humanoid_walk

## 监控命令

```bash
# 查看Montezuma Baseline日志
tail -f dreamerv3_curiosity/runs/baseline_atari_montezuma.log

# 查看最新metrics
tail -1 dreamerv3_curiosity/runs/baseline_atari/montezuma/metrics.jsonl | python3 -m json.tool

# 检查GPU状态
nvidia-smi

# 查看进程
ps -p 3018419 -o pid,etime,%cpu,%mem,cmd
```
