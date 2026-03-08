"""Goal Bank for storing and retrieving high-reward goal states."""

import random
import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax
import embodied.jax.nets as nn


class GoalBank:
  """Stores multiple high-reward goal states with metadata.

  Each goal contains:
  - obs: Observation (game screenshot)
  - latent: (h, z) latent state
  - reward: Reward value
  - metadata: {traj_id, timestep, episode}
  """

  def __init__(self, capacity=100):
    self.capacity = capacity
    self.goals = []

  def add_goal(self, obs, latent, reward, metadata):
    """Add a new goal to the bank.

    Args:
      obs: Observation array (e.g., [64, 64, 3])
      latent: Tuple of (h, z) latent state
      reward: Scalar reward value
      metadata: Dict with {traj_id, timestep, episode}
    """
    goal = {
        'obs': obs,
        'latent': latent,
        'reward': reward,
        'metadata': metadata
    }
    self.goals.append(goal)

    # Keep only top-k by reward
    if len(self.goals) > self.capacity:
      self.goals = sorted(
          self.goals,
          key=lambda x: x['reward'],
          reverse=True
      )[:self.capacity]

  def sample_goal(self, k=1):
    """Sample k goals from the bank.

    Args:
      k: Number of goals to sample

    Returns:
      List of goal dicts
    """
    if len(self.goals) < k:
      return self.goals
    return random.sample(self.goals, k)

  def find_trajectory(self, goal_idx, replay_buffer):
    """Find the historical trajectory that produced this goal.

    Case 1: Goal exists in historical trajectories.
    Returns the action sequence and states leading to the goal.

    Args:
      goal_idx: Index of goal in self.goals
      replay_buffer: Replay buffer containing historical trajectories

    Returns:
      Dict with {observations, actions, latents}
    """
    goal = self.goals[goal_idx]
    traj_id = goal['metadata']['traj_id']
    end_t = goal['metadata']['timestep']

    # Extract sub-trajectory (H steps before goal)
    H = 15  # Configurable
    start_t = max(0, end_t - H)
    trajectory = replay_buffer[traj_id][start_t:end_t]

    return {
        'observations': trajectory['obs'],
        'actions': trajectory['action'],
        'latents': trajectory['latent'],
    }


class GoalConditionedPolicy(nj.Module):
  """Goal-conditioned policy π(a|s, g).

  Takes current state and goal observation as input,
  outputs action distribution.

  Alternative approaches (if performance is poor):
  - Method B: Search/planning (beam search, MCTS)
  - Method D: Attraction field (action selection bias)
  """

  def __init__(self, act_space, config):
    self.act_space = act_space
    self.config = config
    # Use same distribution types as main policy
    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    # Create MLP head for goal-conditioned policy
    # Input will be concatenated [state_feat, goal_feat]
    self.head = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='goal_pol')

  def __call__(self, state_feat, goal_feat):
    """
    Args:
      state_feat: [..., D] current state features
      goal_feat: [..., D] goal state features

    Returns:
      action_dist: Dict of action distributions
    """
    # Concatenate state and goal features
    combined = jnp.concatenate([state_feat, goal_feat], axis=-1)
    # Get action distribution from MLP head
    return self.head(combined, bdims=state_feat.ndim - 1)


class SubgoalDecomposer:
  """Recursively decompose long-horizon goals into subgoals.

  Uses the world model to imagine intermediate states between
  current state and goal state.
  """

  def __init__(self, world_model, max_depth=3):
    self.world_model = world_model
    self.max_depth = max_depth

  def decompose(self, start_state, goal_state, depth=0):
    """Recursively find subgoals between start and goal.

    Args:
      start_state: Current state (h, z)
      goal_state: Goal state (h, z)
      depth: Current recursion depth

    Returns:
      List of subgoal states
    """
    if depth >= self.max_depth:
      return [goal_state]

    # Check if goal is reachable in H steps
    H = 15
    reachable = self._is_reachable(start_state, goal_state, H)

    if reachable:
      return [goal_state]

    # Find midpoint state by imagination
    midpoint = self._find_midpoint(start_state, goal_state, H // 2)

    # Recursively decompose
    subgoals_1 = self.decompose(start_state, midpoint, depth + 1)
    subgoals_2 = self.decompose(midpoint, goal_state, depth + 1)

    return subgoals_1 + subgoals_2

  def _is_reachable(self, start, goal, horizon):
    """Check if goal is reachable from start within horizon steps.

    Uses world model to imagine trajectories and check if any
    trajectory reaches close to goal.

    Args:
      start: Start state (h, z)
      goal: Goal state (h, z)
      horizon: Number of steps to imagine

    Returns:
      Boolean indicating reachability
    """
    # Imagine multiple trajectories from start
    num_samples = 10
    min_distance = float('inf')

    for _ in range(num_samples):
      # Imagine forward for 'horizon' steps
      current_state = start
      for _ in range(horizon):
        # Sample random action
        action = jax.random.categorical(
            jax.random.PRNGKey(0),
            jnp.zeros(self.world_model.act_dim)
        )
        # Step world model
        current_state = self.world_model.step(current_state, action)

      # Compute distance to goal
      distance = self._state_distance(current_state, goal)
      min_distance = min(min_distance, distance)

    # Reachable if minimum distance is below threshold
    return min_distance < 0.3

  def _find_midpoint(self, start, goal, steps):
    """Find intermediate state between start and goal.

    Imagines forward from start for 'steps' steps and selects
    the state closest to goal.

    Args:
      start: Start state (h, z)
      goal: Goal state (h, z)
      steps: Number of steps to imagine

    Returns:
      Midpoint state (h, z)
    """
    # Imagine forward from start
    current_state = start
    best_state = start
    min_distance = self._state_distance(start, goal)

    for _ in range(steps):
      # Sample random action
      action = jax.random.categorical(
          jax.random.PRNGKey(0),
          jnp.zeros(self.world_model.act_dim)
      )
      # Step world model
      current_state = self.world_model.step(current_state, action)

      # Check if closer to goal
      distance = self._state_distance(current_state, goal)
      if distance < min_distance:
        min_distance = distance
        best_state = current_state

    return best_state

  def _state_distance(self, state1, state2):
    """Compute distance between two states in latent space.

    Args:
      state1: State (h, z)
      state2: State (h, z)

    Returns:
      Scalar distance
    """
    # Concatenate h and z
    feat1 = jnp.concatenate([state1[0], state1[1].flatten()], axis=-1)
    feat2 = jnp.concatenate([state2[0], state2[1].flatten()], axis=-1)

    # Cosine distance
    norm1 = jnp.linalg.norm(feat1) + 1e-8
    norm2 = jnp.linalg.norm(feat2) + 1e-8
    similarity = jnp.dot(feat1, feat2) / (norm1 * norm2)

    return 1.0 - similarity
