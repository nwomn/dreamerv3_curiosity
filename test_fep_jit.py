"""Test JIT compatibility of FEP components."""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '.')

from dreamerv3 import fep_components


def test_goal_bank_jit():
  """Test GoalBank is JIT compatible."""
  print("Testing GoalBank JIT compatibility...")

  goal_bank = fep_components.GoalBank(capacity=100, latent_dim=64, name='test_goal_bank')
  state = goal_bank.initial_state(batch_size=4)

  # Create dummy data
  latents = jnp.ones((4, 10, 64))
  rewards = jnp.ones((4, 10))
  dones = jnp.zeros((4, 10)).at[:, -1].set(1.0)

  # JIT compile
  @jax.jit
  def update(state, latents, rewards, dones):
    return goal_bank(state, latents, rewards, dones)

  # Should not raise TracerError
  try:
    new_state = update(state, latents, rewards, dones)
    print("✓ GoalBank update JIT compiled successfully")

    # Check state shape
    assert new_state['goals'].shape == (100, 64), f"Expected (100, 64), got {new_state['goals'].shape}"
    assert new_state['valid_mask'].sum() > 0, "No valid goals stored"
    print(f"✓ GoalBank stored {new_state['valid_mask'].sum()} goals")

  except Exception as e:
    print(f"✗ JIT compilation failed: {e}")
    return False

  return True


def test_goal_sampling():
  """Test goal sampling."""
  print("\nTesting GoalBank sampling...")

  goal_bank = fep_components.GoalBank(capacity=100, latent_dim=64, name='test_goal_bank2')
  state = goal_bank.initial_state(batch_size=4)

  # Add some goals
  latents = jnp.ones((4, 10, 64))
  rewards = jnp.ones((4, 10))
  dones = jnp.zeros((4, 10)).at[:, -1].set(1.0)
  state = goal_bank(state, latents, rewards, dones)

  # Sample goals
  @jax.jit
  def sample(state, key):
    return goal_bank.sample_goals(state, k=8, key=key)

  try:
    key = jax.random.PRNGKey(0)
    goals = sample(state, key)
    assert goals.shape == (8, 64), f"Expected (8, 64), got {goals.shape}"
    print(f"✓ Sampled {goals.shape[0]} goals successfully")
  except Exception as e:
    print(f"✗ Sampling failed: {e}")
    return False

  return True


if __name__ == '__main__':
  success = True
  success &= test_goal_bank_jit()
  success &= test_goal_sampling()

  if success:
    print("\n✓ All tests passed!")
  else:
    print("\n✗ Some tests failed")
    sys.exit(1)
