# Pong 第二轮实验 (Run 2)

**启动时间**: 2026-03-08 21:32  
**目的**: 验证 Run 1 结果的可重复性

---

## 实验配置

### Run 1 (已完成)
- **Seed**: 0 (默认)
- **Steps**: 4,000,000
- **结果**:
  - Baseline: 14.12 分, reward_rate 0.0198
  - FEP: 11.92 分, reward_rate 0.0204 ✅
  - **发现**: FEP reward_rate 更高但最终得分更低

### Run 2 (进行中)
- **Seed**: 42 (不同于 Run 1)
- **Steps**: 4,000,000
- **GPU**: Baseline=0, FEP=3
- **PID**: Baseline=2302483, FEP=2302484
- **预计完成**: ~3 小时后 (约 00:30)

---

## 验证目标

1. **Baseline 是否稳定优于 FEP？**
   - Run 1: Baseline 14.12 vs FEP 11.92 (+2.20)
   - Run 2: 待验证

2. **FEP reward_rate 是否持续更高？**
   - Run 1: FEP 0.0204 vs Baseline 0.0198 (+3.3%)
   - Run 2: 待验证

3. **性能差距是否稳定？**
   - 如果 Run 2 结果相似，说明结论可靠
   - 如果差异很大，需要更多实验

---

## 监控命令

```bash
# 查看进程
ps aux | grep 'python.*main.py' | grep -v grep

# 查看日志
tail -f runs/baseline_atari_pong_run2.log
tail -f runs/fep_atari_pong_run2.log

# 查看最新指标
tail -1 runs/baseline_atari/pong_run2/metrics.jsonl | python3 -m json.tool
tail -1 runs/fep_atari/pong_run2/metrics.jsonl | python3 -m json.tool

# GPU 状态
nvidia-smi
```

---

## 文件位置

- **Baseline Run 2**:
  - 日志: `runs/baseline_atari_pong_run2.log`
  - Metrics: `runs/baseline_atari/pong_run2/metrics.jsonl`
  - Checkpoints: `runs/baseline_atari/pong_run2/ckpt/`

- **FEP Run 2**:
  - 日志: `runs/fep_atari_pong_run2.log`
  - Metrics: `runs/fep_atari/pong_run2/metrics.jsonl`
  - Checkpoints: `runs/fep_atari/pong_run2/ckpt/`

---

**状态**: ✅ 训练中  
**GPU 使用**: GPU 0 (61GB), GPU 3 (61GB)  
**下次检查**: 22:30 (1小时后)
