# Curiosity-Driven Exploration Guide

This guide explains how to use the modular curiosity mechanism in DreamerV3 for benchmark testing.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration System](#configuration-system)
- [Benchmark Presets](#benchmark-presets)
- [Parameter Tuning](#parameter-tuning)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Running with Default Curiosity Settings

Each benchmark comes with recommended curiosity settings:

```bash
# Atari with curiosity (moderate exploration)
python dreamerv3/main.py --configs atari --task atari_pong

# DMLab with curiosity (aggressive exploration)
python dreamerv3/main.py --configs dmlab --task dmlab_explore_goal_locations_small

# DMC without curiosity (dense reward)
python dreamerv3/main.py --configs dmc_vision --task dmc_walker_walk
```

### Disabling Curiosity for Baseline Comparison

```bash
# Disable curiosity for any benchmark
python dreamerv3/main.py --configs atari --agent.curiosity.enabled False
```

### Enabling Curiosity for Benchmarks that Have it Disabled by Default

```bash
# Enable curiosity for DMC
python dreamerv3/main.py --configs dmc_vision --agent.curiosity.enabled True
```

---

## Configuration System

### Nested Configuration Structure

Curiosity settings are now organized under `agent.curiosity`:

```yaml
agent:
  curiosity:
    enabled: True       # Enable/disable curiosity mechanism
    samples: 10         # Number of actions to sample per step
    alpha: 0.001       # EMA smoothing factor for dynamic threshold
    std_scale: 1.0     # Standard deviation multiplier for threshold
```

### Configuration Hierarchy

Configurations are applied in this order (later overrides earlier):

1. **Global defaults** (configs.yaml `defaults` section)
2. **Benchmark presets** (configs.yaml benchmark sections)
3. **Command-line overrides** (`--agent.curiosity.enabled False`)

---

## Benchmark Presets

### Pre-configured Benchmarks

| Benchmark | Curiosity | samples | std_scale | alpha | Rationale |
|-----------|-----------|---------|-----------|-------|-----------|
| **atari** | ✅ Enabled | 10 | 1.5 | 0.001 | Balanced exploration for mixed sparse/dense rewards |
| **dmlab** | ✅ Enabled | 15 | 2.0 | 0.001 | Aggressive exploration for 3D navigation |
| **crafter** | ✅ Enabled | 10 | 1.2 | 0.0005 | Long-term exploration, slower adaptation |
| **minecraft** | ✅ Enabled | 10 | 1.5 | 0.001 | Moderate exploration for open-world |
| **procgen** | ✅ Enabled | 10 | 1.3 | 0.001 | Adaptive exploration for procedural generation |
| **atari100k** | ✅ Enabled | 5 | 1.2 | 0.001 | Data-efficient: fewer samples |
| **loconav** | ✅ Enabled | 10 | 1.5 | 0.001 | Navigation benefits from exploration |
| **dmc_vision** | ❌ Disabled | - | - | - | Continuous control, dense rewards |
| **dmc_proprio** | ❌ Disabled | - | - | - | Dense rewards, no exploration needed |
| **bsuite** | ❌ Disabled | - | - | - | Behavioral test suite |

### When to Use Curiosity

**Enable for:**
- Sparse reward environments (Montezuma's Revenge, exploration tasks)
- Open-world games (Minecraft, Crafter)
- Navigation tasks (DMLab, LocoNav)
- Procedurally generated environments (ProcGen)

**Disable for:**
- Dense reward tasks (most DMC tasks)
- Well-understood environments (simple Atari games like Pong)
- Data-limited scenarios where computational efficiency matters
- Baseline comparisons

---

## Parameter Tuning

### Core Parameters Explained

#### 1. **enabled** (Boolean)
Global toggle for curiosity mechanism.

```bash
--agent.curiosity.enabled True   # Enable
--agent.curiosity.enabled False  # Disable
```

#### 2. **samples** (Integer, default: 10)
Number of actions sampled per step to evaluate uncertainty.

- **Higher values** (15-20): More comprehensive uncertainty estimation, higher computational cost
- **Lower values** (5-8): Faster, good for data-efficient settings
- **Recommendation**: 10 for most tasks, 15 for complex 3D environments, 5 for atari100k

```bash
--agent.curiosity.samples 15
```

#### 3. **std_scale** (Float, default: 1.0)
Threshold sensitivity: `threshold = mean + std_scale × std`

- **Higher values** (1.5-2.0): More exploration, higher threshold to exceed
- **Lower values** (0.5-1.0): Less exploration, easier to trigger
- **Recommendation**: 1.5 for balanced, 2.0 for aggressive, 1.0 for conservative

```bash
--agent.curiosity.std_scale 2.0
```

#### 4. **alpha** (Float, default: 0.001)
EMA smoothing factor for running statistics: `mean_new = mean_old + alpha × (value - mean_old)`

- **Higher values** (0.01): Faster adaptation to new uncertainty levels
- **Lower values** (0.0001): Slower, more stable threshold
- **Recommendation**: 0.001 for most tasks, 0.0005 for long-term exploration (Crafter, Minecraft)

```bash
--agent.curiosity.alpha 0.0005
```

---

## Advanced Usage

### Ablation Studies

#### Curiosity On vs Off

```bash
# Experiment group (with curiosity)
python dreamerv3/main.py --configs atari --task atari_montezuma_revenge \
  --logdir logs/montezuma_curiosity

# Control group (without curiosity)
python dreamerv3/main.py --configs atari --task atari_montezuma_revenge \
  --agent.curiosity.enabled False \
  --logdir logs/montezuma_baseline
```

#### Parameter Sensitivity Analysis

Test different `std_scale` values:

```bash
for scale in 0.5 1.0 1.5 2.0; do
  python dreamerv3/main.py --configs dmlab \
    --agent.curiosity.std_scale $scale \
    --logdir logs/dmlab_scale_$scale
done
```

### Custom Configurations

Create your own curiosity preset:

```bash
python dreamerv3/main.py \
  --configs atari \
  --task atari_venture \
  --agent.curiosity.enabled True \
  --agent.curiosity.samples 20 \
  --agent.curiosity.std_scale 2.5 \
  --agent.curiosity.alpha 0.0005
```

### Combining with Other Configs

```bash
# Small model + curiosity
python dreamerv3/main.py --configs atari size50m \
  --agent.curiosity.std_scale 1.8

# Debug mode with curiosity disabled
python dreamerv3/main.py --configs debug \
  --agent.curiosity.enabled False
```

---

## Troubleshooting

### Issue: Agent explores too much / doesn't learn

**Solution**: Decrease `std_scale` or disable curiosity.

```bash
--agent.curiosity.std_scale 0.8
```

### Issue: Agent doesn't explore enough / stuck in local optima

**Solution**: Increase `std_scale` or `samples`.

```bash
--agent.curiosity.std_scale 2.0 --agent.curiosity.samples 15
```

### Issue: Training is too slow

**Solution**: Reduce `samples` or disable curiosity.

```bash
--agent.curiosity.samples 5
```

### Issue: Threshold adapts too quickly

**Solution**: Decrease `alpha` for more stable thresholds.

```bash
--agent.curiosity.alpha 0.0001
```

### Issue: Threshold doesn't adapt to environment changes

**Solution**: Increase `alpha` for faster adaptation.

```bash
--agent.curiosity.alpha 0.01
```

---

## Implementation Details

### How It Works

1. **Action Sampling**: At each step, sample `samples` actions from policy distribution
2. **Uncertainty Evaluation**: Compute mean entropy of RSSM prior distribution for all actions
3. **Dynamic Threshold**: Update running mean/variance using EMA, threshold = mean + std_scale × std
4. **Exploration Decision**: If current entropy > threshold, select action with max entropy; otherwise sample from policy

### Performance Impact

- **Computation**: ~10-20% overhead when enabled (depends on `samples`)
- **Memory**: Minimal (<1 MB for curiosity state)
- **Recommended**: Enable for sparse reward tasks, disable for dense reward or data-efficient settings

---

## Examples by Use Case

### Sparse Reward Exploration

```bash
# Montezuma's Revenge (notoriously sparse reward)
python dreamerv3/main.py --configs atari \
  --task atari_montezuma_revenge \
  --agent.curiosity.std_scale 2.0 \
  --agent.curiosity.samples 15
```

### Data-Efficient Learning

```bash
# Atari with 100k steps limit
python dreamerv3/main.py --configs atari100k \
  --task atari100k_breakout
  # Uses preset: samples=5, std_scale=1.2
```

### Dense Reward Baseline

```bash
# DMC Walker (dense reward, no curiosity needed)
python dreamerv3/main.py --configs dmc_vision
  # Curiosity automatically disabled by preset
```

### Long-Term Exploration

```bash
# Minecraft diamond collection
python dreamerv3/main.py --configs minecraft \
  --agent.curiosity.alpha 0.0005
  # Slower threshold adaptation for long-term goals
```

---

## Configuration Quick Reference

```bash
# Template for custom experiments
python dreamerv3/main.py \
  --configs <benchmark> \
  --task <task_name> \
  --agent.curiosity.enabled <True|False> \
  --agent.curiosity.samples <5-20> \
  --agent.curiosity.std_scale <0.5-2.5> \
  --agent.curiosity.alpha <0.0001-0.01> \
  --logdir <output_directory>
```

---

## Version Information

- **DreamerV3 Version**: 3.3.1
- **Curiosity Mechanism**: Dynamic Threshold with Entropy-Based Uncertainty
- **Implementation**: Stateless, JAX-compatible
- **Last Updated**: 2025-11-21

For bugs or feature requests, see: `BUG_FIX_SUMMARY.md` and `REFACTORING_PLAN.md`
