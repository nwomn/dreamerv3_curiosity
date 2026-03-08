"""FEP (Free Energy Principle) components for DreamerV3.

This module contains JIT-compatible implementations of:
- GoalBank: Store and sample goal states
- GoalConditionedPolicy: Policy conditioned on goal state
- SubgoalDecomposer: Recursive subgoal decomposition (JIT-external)

All components in this file are designed to work with JAX JIT compilation.
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax
import embodied.jax.nets as nn


class GoalBank(nj.Module):
  """Store and sample goal states (JIT-compatible).

  Uses fixed-size storage with masking to ensure JIT compatibility.
  Goals are stored in a circular buffer with a validity mask.
  """

  def __init__(self, capacity=1000, latent_dim=1024, name='goal_bank'):
    self.capacity = capacity
    self.latent_dim = latent_dim
    self._name = name

  def initial_state(self, batch_size=1):
    """Initialize goal bank state.

    Returns:
      Dict with:
        - goals: (capacity, D) goal latents
        - rewards: (capacity,) reward values
        - valid_mask: (capacity,) boolean mask for valid goals
        - write_ptr: scalar write pointer
    """
    return {
        'goals': jnp.zeros((self.capacity, self.latent_dim), jnp.float32),
        'rewards': jnp.zeros(self.capacity, jnp.float32),
        'valid_mask': jnp.zeros(self.capacity, jnp.bool_),
        'write_ptr': jnp.array(0, jnp.int32),
    }

  def __call__(self, state, latents, rewards, dones):
    """Update goal bank with new latents (JIT-compatible).

    Args:
      state: Current goal bank state
      latents: (B, T, D) latent states
      rewards: (B, T) rewards
      dones: (B, T) episode termination flags

    Returns:
      Updated goal bank state
    """
    B, T, D = latents.shape

    # Flatten batch and time dimensions
    flat_latents = latents.reshape(-1, D)  # (B*T, D)
    flat_rewards = rewards.reshape(-1)     # (B*T,)
    flat_dones = dones.reshape(-1)         # (B*T,)

    # Filter: only keep states with done=True (goal states)
    # Use masking instead of dynamic filtering
    is_goal = flat_dones > 0.5  # (B*T,)

    # Write up to K goals (K is fixed for JIT compatibility)
    K = 32  # Maximum number of goals to write per call
    write_indices = jnp.arange(K)

    # Loop through and write goals (using fori_loop for JIT compatibility)
    def write_one(i, carry_state):
      idx = (carry_state['write_ptr'] + i) % self.capacity
      # Only write if i < B*T and is_goal[i] is True
      should_write = (i < B * T) & is_goal[i]

      new_goals = carry_state['goals'].at[idx].set(
          jnp.where(should_write, flat_latents[i], carry_state['goals'][idx])
      )
      new_rewards = carry_state['rewards'].at[idx].set(
          jnp.where(should_write, flat_rewards[i], carry_state['rewards'][idx])
      )
      new_mask = carry_state['valid_mask'].at[idx].set(
          should_write | carry_state['valid_mask'][idx]
      )

      return {
          **carry_state,
          'goals': new_goals,
          'rewards': new_rewards,
          'valid_mask': new_mask,
      }

    new_state = jax.lax.fori_loop(0, K, write_one, state)
    new_state['write_ptr'] = (state['write_ptr'] + K) % self.capacity

    return new_state

  def sample_goals(self, state, k=8, key=None):
    """Sample k goals from the bank (JIT-compatible).

    Args:
      state: Goal bank state
      k: Number of goals to sample (must be static)
      key: JAX random key (if None, will use nj.seed())

    Returns:
      (k, D) goal latents
    """
    # Only sample from valid goals
    valid_indices = jnp.where(
        state['valid_mask'],
        jnp.arange(self.capacity),
        -1  # Invalid index marker
    )

    # Randomly sample k indices
    if key is None:
      key = nj.seed()
    sampled_idx = jax.random.choice(
        key, valid_indices, shape=(k,), replace=True
    )

    # Handle invalid indices (replace with first valid goal)
    first_valid = jnp.argmax(state['valid_mask'])
    sampled_idx = jnp.where(sampled_idx >= 0, sampled_idx, first_valid)

    return state['goals'][sampled_idx]



class GoalConditionedPolicy(nj.Module):
  """Goal-conditioned policy π(a|s, g).

  Takes current state and goal state as input, outputs action distribution.
  Uses the same MLP architecture as the main policy.
  """

  def __init__(self, act_space, config, name='goal_cond_policy'):
    self.act_space = act_space
    self.config = config
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.head = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='head')

  def __call__(self, state_feat, goal_feat):
    """Compute action distribution given state and goal.

    Args:
      state_feat: (..., D) current state features
      goal_feat: (..., D) goal state features

    Returns:
      Action distribution dict
    """
    combined = jnp.concatenate([state_feat, goal_feat], axis=-1)
    return self.head(combined, bdims=state_feat.ndim - 1)


class ExpectedFreeEnergy(nj.Module):
  """Compute Expected Free Energy (EFE) for action sequences.

  EFE = Epistemic Value + Pragmatic Value

  - Epistemic Value: Information gain (exploration)
  - Pragmatic Value: Expected reward (exploitation)

  This is JIT-compatible and integrates with DreamerV3's world model.
  """

  def __init__(self, config, name='efe'):
    self.config = config
    self._name = name

  def __call__(self, world_model, start_state, actions, goals=None):
    """Compute EFE for a sequence of imagined actions.

    Args:
      world_model: DreamerV3 world model (for imagination)
      start_state: (B, D) starting latent state
      actions: (B, H, A) action sequence (H = horizon)
      goals: Optional (B, D) goal states for pragmatic value

    Returns:
      efe: (B,) Expected Free Energy values (lower is better)
    """
    B, H, A = actions.shape

    # Imagine trajectory using world model
    # start_state should be a dict with 'deter' and 'stoch'
    imag_states = self._imagine_trajectory(world_model, start_state, actions)

    # Compute epistemic value (information gain)
    epistemic = self._epistemic_value(imag_states)

    # Compute pragmatic value (expected reward)
    pragmatic = self._pragmatic_value(world_model, imag_states, goals)

    # Combine with weights from config
    alpha = self.config.fep.get('alpha_max', 0.1)  # Epistemic weight
    beta = self.config.fep.get('beta_max', 0.05)   # Pragmatic weight

    # EFE = -epistemic_value - pragmatic_value
    # (we want to maximize both, so minimize negative)
    efe = -alpha * epistemic - beta * pragmatic

    return efe

  def _imagine_trajectory(self, world_model, start_state, actions):
    """Imagine trajectory by rolling out actions in world model.

    Args:
      world_model: DreamerV3 RSSM
      start_state: Dict with 'deter' and 'stoch'
      actions: (B, H, A) actions

    Returns:
      states: Dict with 'deter' (B, H, D1) and 'stoch' (B, H, D2)
    """
    B, H, A = actions.shape

    # Initialize trajectory storage
    deter_list = []
    stoch_list = []

    # Current state
    state = start_state

    # Roll out trajectory (use scan for efficiency)
    def step_fn(carry_state, action):
      # Predict next state given action
      next_state = world_model.img_step(carry_state, action)
      return next_state, next_state

    # Scan over time dimension
    _, imag_states = jax.lax.scan(
        step_fn,
        start_state,
        jnp.moveaxis(actions, 1, 0)  # (H, B, A)
    )

    return imag_states

  def _epistemic_value(self, states):
    """Compute epistemic value (information gain).

    Measures uncertainty reduction. Higher uncertainty = higher epistemic value.
    We use the entropy of the stochastic state distribution.

    Args:
      states: Dict with 'stoch' (B, H, D) or (H, B, D)

    Returns:
      epistemic: (B,) epistemic value (higher = more informative)
    """
    # Get stochastic state
    stoch = states.get('stoch', states.get('z', None))

    if stoch is None:
      # Fallback: use deterministic state variance
      deter = states.get('deter', states.get('h', None))
      if deter.ndim == 3:  # (H, B, D) or (B, H, D)
        # Check if first dim is smaller (likely H from scan)
        if deter.shape[0] < deter.shape[1]:
          deter = jnp.moveaxis(deter, 0, 1)  # (H, B, D) -> (B, H, D)
      # Compute variance across time as proxy for uncertainty
      epistemic = jnp.var(deter, axis=1).mean(axis=-1)  # (B,)
      return epistemic

    # Handle different shapes from scan
    if stoch.ndim == 3:  # (H, B, D) or (B, H, D)
      # Check if first dim is smaller (likely H from scan)
      if stoch.shape[0] < stoch.shape[1]:
        stoch = jnp.moveaxis(stoch, 0, 1)  # (H, B, D) -> (B, H, D)

    B, H, D = stoch.shape

    # For simplicity, use variance across time as epistemic value
    # This measures how much the state changes over the trajectory
    epistemic = jnp.var(stoch, axis=1).mean(axis=-1)  # (B,)

    return epistemic

  def _pragmatic_value(self, world_model, states, goals=None):
    """Compute pragmatic value (expected reward).

    Args:
      world_model: DreamerV3 world model (has reward predictor)
      states: Imagined states dict
      goals: Optional (B, D) goal states

    Returns:
      pragmatic: (B,) pragmatic value (higher = better)
    """
    # Get state features for reward prediction
    stoch = states.get('stoch', states.get('z', None))
    deter = states.get('deter', states.get('h', None))

    # Handle scan output shape (H, B, D) -> (B, H, D)
    if stoch.ndim == 3 and stoch.shape[0] < stoch.shape[1]:
      stoch = jnp.moveaxis(stoch, 0, 1)
    if deter.ndim == 3 and deter.shape[0] < deter.shape[1]:
      deter = jnp.moveaxis(deter, 0, 1)

    B, H = stoch.shape[:2]

    # Concatenate deter and stoch to get full state
    # This matches DreamerV3's feature representation
    state_feat = jnp.concatenate([deter, stoch], axis=-1)  # (B, H, D)

    # Predict rewards for imagined trajectory
    # Reshape to (B*H, D) for reward head
    state_feat_flat = state_feat.reshape(B * H, -1)

    # Use world model's reward predictor
    # Note: world_model.heads['reward'] expects features
    reward_dist = world_model.heads['reward'](state_feat_flat)
    predicted_rewards = reward_dist.mean()  # (B*H,)
    predicted_rewards = predicted_rewards.reshape(B, H)  # (B, H)

    # Sum rewards over horizon (discounted)
    gamma = 0.99
    discount = jnp.power(gamma, jnp.arange(H))  # (H,)
    pragmatic = jnp.sum(predicted_rewards * discount[None, :], axis=1)  # (B,)

    # If goals provided, add goal-reaching bonus
    if goals is not None:
      # Compute distance to goal at final state
      final_state = state_feat[:, -1, :]  # (B, D)
      goal_dist = jnp.sum((final_state - goals) ** 2, axis=-1)  # (B,)
      goal_bonus = -goal_dist  # Negative distance as bonus
      pragmatic = pragmatic + 0.1 * goal_bonus

    return pragmatic


class SubgoalDecomposer:
  """Recursively decompose long-horizon goals into subgoals.

  NOTE: This runs OUTSIDE of JIT context, so can use Python control flow.

  Uses world model to:
  1. Check if goal is directly reachable
  2. If not, find intermediate subgoals
  3. Recursively decompose until all subgoals are reachable

  This enables hierarchical planning for long-horizon tasks.
  """

  def __init__(self, world_model, max_depth=3, horizon=15):
    """Initialize subgoal decomposer.

    Args:
      world_model: DreamerV3 RSSM for reachability checking
      max_depth: Maximum recursion depth
      horizon: Planning horizon for reachability check
    """
    self.world_model = world_model
    self.max_depth = max_depth
    self.horizon = horizon

  def decompose(self, start_state, goal_state, depth=0):
    """Recursively find subgoals between start and goal.

    Args:
      start_state: Current state dict with 'deter' and 'stoch'
      goal_state: Goal state dict with 'deter' and 'stoch'
      depth: Current recursion depth

    Returns:
      List of subgoal states (including final goal)
    """
    # Base case 1: Max depth reached
    if depth >= self.max_depth:
      return [goal_state]

    # Base case 2: Goal is directly reachable
    if self._is_reachable(start_state, goal_state, self.horizon):
      return [goal_state]

    # Recursive case: Find midpoint and decompose
    midpoint = self._find_midpoint(start_state, goal_state)

    # Decompose start -> midpoint
    subgoals_first = self.decompose(start_state, midpoint, depth + 1)

    # Decompose midpoint -> goal
    subgoals_second = self.decompose(midpoint, goal_state, depth + 1)

    # Combine (avoid duplicating midpoint)
    return subgoals_first + subgoals_second

  def _is_reachable(self, start, goal, horizon):
    """Check if goal is reachable from start within horizon steps.

    Uses world model to imagine trajectory and check if we can get close
    to the goal state.

    Args:
      start: Start state dict
      goal: Goal state dict
      horizon: Number of steps to check

    Returns:
      bool: True if goal appears reachable
    """
    # Extract features
    start_feat = self._state_to_feature(start)
    goal_feat = self._state_to_feature(goal)

    # Compute distance threshold
    # If states are close enough, consider reachable
    distance = jnp.sqrt(jnp.sum((start_feat - goal_feat) ** 2))

    # Heuristic: reachable if distance is small relative to horizon
    # This is a simple heuristic; more sophisticated methods could use
    # actual trajectory imagination
    threshold = 5.0 * jnp.sqrt(horizon)  # Scale with horizon

    return float(distance) < threshold

  def _find_midpoint(self, start, goal):
    """Find intermediate state between start and goal.

    Uses linear interpolation in latent space as a simple heuristic.
    More sophisticated methods could use world model imagination.

    Args:
      start: Start state dict
      goal: Goal state dict

    Returns:
      Midpoint state dict
    """
    # Linear interpolation in latent space
    midpoint = {}

    for key in ['deter', 'stoch']:
      if key in start and key in goal:
        # Interpolate at 0.5 (midpoint)
        midpoint[key] = 0.5 * start[key] + 0.5 * goal[key]

    return midpoint

  def _state_to_feature(self, state):
    """Convert state dict to feature vector.

    Args:
      state: State dict with 'deter' and 'stoch'

    Returns:
      Feature vector (concatenated deter and stoch)
    """
    deter = state.get('deter', jnp.array([]))
    stoch = state.get('stoch', jnp.array([]))

    # Handle batch dimensions
    if deter.ndim > 1:
      deter = deter.reshape(-1)
    if stoch.ndim > 1:
      stoch = stoch.reshape(-1)

    return jnp.concatenate([deter, stoch])

  def plan_to_goal(self, start_state, goal_state, policy):
    """Plan a sequence of actions to reach goal via subgoals.

    This is a high-level planning function that:
    1. Decomposes goal into subgoals
    2. Plans to each subgoal sequentially
    3. Returns combined action sequence

    Args:
      start_state: Current state
      goal_state: Target goal state
      policy: Goal-conditioned policy π(a|s,g)

    Returns:
      List of actions to reach goal
    """
    # Decompose into subgoals
    subgoals = self.decompose(start_state, goal_state)

    # Plan to each subgoal
    actions = []
    current_state = start_state

    for subgoal in subgoals:
      # Use goal-conditioned policy to generate actions
      # This is a simplified version; actual implementation would
      # use world model imagination
      subgoal_actions = self._plan_to_subgoal(
          current_state, subgoal, policy
      )
      actions.extend(subgoal_actions)

      # Update current state (would use world model in practice)
      current_state = subgoal

    return actions

  def _plan_to_subgoal(self, start, subgoal, policy, steps=10):
    """Plan actions to reach a single subgoal.

    Args:
      start: Start state
      subgoal: Target subgoal
      policy: Goal-conditioned policy
      steps: Number of planning steps

    Returns:
      List of actions
    """
    # Placeholder: In practice, this would use world model imagination
    # and goal-conditioned policy to generate action sequence
    # For now, return empty list (to be implemented when integrated)
    return []


