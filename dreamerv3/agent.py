import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from . import fep_components


f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc') # 这里通过self.enc调用rssm.Encoder的__init__方法
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn') # 这里通过self.dyn调用rssm.RSSM的__init__方法
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec') # 这里通过self.dec调用rssm.Decoder的__init__方法

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # FEP modules (only active when fep.enabled=True)
    self.fep_enabled = config.fep.enabled
    if self.fep_enabled:
      scalar = elements.Space(np.float32, ())
      self.info_gain = embodied.jax.MLPHead(
          scalar, **config.info_gain_head, name='info_gain')
      rssm_cfg = config.dyn[config.dyn.typ]
      feat_dim = rssm_cfg.deter + rssm_cfg.stoch * rssm_cfg.classes
      self.goal_imaginator = GoalImaginator(feat_dim, config)
      self.goal_imaginator.init_variables()
      self.fep_scheduler = FEPScheduler(config)
      self.fep_scheduler.init_variables()

      # Goal Bank for Level 2 Imagination
      self.goal_bank = fep_components.GoalBank(capacity=100, name='goal_bank')

      # Goal-conditioned policy for Case 2
      self.goal_cond_policy = fep_components.GoalConditionedPolicy(
          self.act_space, config, name='goal_cond_policy')

      # Subgoal decomposer for Case 2 (Phase 5)
      self.subgoal_decomposer = fep_components.SubgoalDecomposer(
          world_model=self.dyn,
          max_depth=config.fep.get('subgoal_max_depth', 3),
          horizon=config.fep.get('subgoal_horizon', 15))


    modules = [self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if self.fep_enabled:
      modules.append(self.info_gain)
      modules.append(self.goal_cond_policy)
    self.modules = modules
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt') # 这里通过self.opt调用embodied.jax.Optimizer的__init__方法

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if not self.fep_enabled:
      scales.pop('info_gain', None)
      scales.pop('goal_policy', None)
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'): # 这个函数的目的是根据输入的obs和carry，输出act和out
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw) # 这里通过self.enc调用rssm.Encoder的__call__方法
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw) # 这里通过self.dyn调用rssm.RSSM的observe方法
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw) # 这里通过self.dec调用rssm.Decoder的__call__方法
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True) # 这里通过embodied.jax.Optimizer调用了Agent的loss方法
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training): # 这个函数的目的是根据输入的obs和carry，输出loss和metrics
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # FEP Phase 2: Train info_gain head and update FEP state
    if self.fep_enabled:
      # Train info_gain head to predict CUMULATIVE KL over H steps
      ig_inp = sg(self.feat2tensor(repfeat))  # [B, T, D], stop gradient
      dyn_loss = sg(losses['dyn'])  # [B, T], stop gradient from KL

      # Compute cumulative KL target: sum of KL from t to t+H
      H = self.config.imag_length
      cumulative_kl = jnp.zeros_like(dyn_loss)
      for h in range(H):
        shifted = jnp.roll(dyn_loss, -h, axis=1)
        # Mask out invalid positions (beyond sequence end)
        mask = (jnp.arange(T)[None, :] + h < T).astype(f32)
        cumulative_kl += shifted * mask

      # Normalize to have mean=0, std=1 (so head learns relative values)
      ig_target = (cumulative_kl - cumulative_kl.mean()) / (cumulative_kl.std() + 1e-8)
      ig_target = sg(ig_target)

      losses['info_gain'] = self.info_gain(ig_inp, 2).loss(ig_target)

      # Update goal imaginator with current features and rewards
      self.goal_imaginator.update_goal(
          sg(self.feat2tensor(repfeat)), obs['reward'])

      # FEP Phase 3: Train goal-conditioned policy
      # Always compute forward pass (required for JIT parameter creation),
      # use masking to zero out loss when no high-reward states exist.
      state_feat = sg(self.feat2tensor(repfeat))  # [B, T, D]

      flat_rew = obs['reward'].reshape(B * T)
      flat_feat = state_feat.reshape(B * T, -1)
      flat_prevact = jax.tree.map(lambda x: x.reshape(B * T, *x.shape[2:]), prevact)

      topk = 8  # Static value required by JIT
      H_lookback = 5
      sorted_idx = jnp.argsort(flat_rew)
      topk_idx = sorted_idx[-topk:]
      topk_rew = flat_rew[topk_idx]

      # Get goal features and lookback trajectories
      goal_feats = flat_feat[topk_idx]  # [topk, D]
      start_indices = jnp.maximum(topk_idx - H_lookback, 0)
      offsets = jnp.arange(H_lookback)[None, :]
      traj_indices = jnp.clip(start_indices[:, None] + offsets, 0, B * T - 1)

      traj_feats = flat_feat[traj_indices.reshape(-1)].reshape(
          topk, H_lookback, -1)
      traj_acts = jax.tree.map(
          lambda x: x[traj_indices.reshape(-1)].reshape(topk, H_lookback, *x.shape[1:]),
          flat_prevact)
      goal_feats_repeated = jnp.repeat(
          goal_feats[:, None, :], H_lookback, axis=1)

      state_feat_flat = traj_feats.reshape(topk * H_lookback, -1)
      goal_feat_flat = goal_feats_repeated.reshape(topk * H_lookback, -1)

      # Always run forward pass so parameters are created under JIT
      action_pred = self.goal_cond_policy(
          sg(state_feat_flat), sg(goal_feat_flat))

      # Mask: only count high-reward goals
      reward_threshold = 0.5
      high_rew_mask = topk_rew > reward_threshold  # [topk]
      mask = jnp.repeat(high_rew_mask, H_lookback)  # [topk * H_lookback]

      gp_losses = []
      for act_key in self.act_space:
        act_target = traj_acts[act_key].reshape(
            topk * H_lookback, *traj_acts[act_key].shape[2:])
        loss = -action_pred[act_key].logp(sg(act_target))
        loss = loss * mask
        gp_losses.append(loss.sum() / (mask.sum() + 1e-8))

      losses['goal_policy'] = jnp.stack(gp_losses).mean()

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)

    # Compute reward with optional FEP augmentation
    img_rew = self.rew(inp, 2).pred()
    if self.fep_enabled:
      # Compute prior entropy from imagination trajectory (skip first step
      # which comes from observe-stage posterior, use only pure prior logits)
      img_prior_ent = self.dyn._dist(imgfeat['logit'][:, 1:]).entropy().mean()

      # Update scheduler with forward-looking prior entropy
      fep_alpha, fep_beta, sched_mets = self.fep_scheduler.update_and_get(
          img_prior_ent, training)
      metrics.update(sched_mets)

      # Info gain: predicted KL along imagined trajectory
      img_info_gain = self.info_gain(sg(inp), 2).pred()
      # Goal proximity: cosine similarity to goal state
      img_goal_prox = self.goal_imaginator.proximity(sg(inp))
      # Augmented reward = reward + alpha * info_gain + beta * goal_proximity
      augmented_rew = img_rew + fep_alpha * img_info_gain + fep_beta * img_goal_prox
      metrics['fep/alpha'] = fep_alpha
      metrics['fep/beta'] = fep_beta
      metrics['fep/img_info_gain'] = img_info_gain.mean()
      metrics['fep/img_goal_prox'] = img_goal_prox.mean()
      metrics['fep/reward_seen'] = self.goal_imaginator.reward_seen.read()
    else:
      augmented_rew = img_rew

    los, imgloss_out, mets = imag_loss(
        imgact,
        augmented_rew,
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def find_similar_trajectories(self, goal_obs, replay_buffer, top_k=5):
    """Find trajectories that reach states similar to goal_obs.

    Uses hybrid search (Case 1):
    - Stage 1: Latent space coarse filter (fast)
    - Stage 2: Observation space precise matching (accurate, SSIM)

    Alternative approaches (if performance is poor):
    - Method B: Search/planning (beam search, MCTS)
    - Method D: Attraction field (action selection bias)

    Args:
      goal_obs: Goal observation (e.g., [64, 64, 3])
      replay_buffer: Replay buffer containing historical trajectories
      top_k: Number of top similar trajectories to return

    Returns:
      List of (traj_id, timestep, similarity) tuples
    """
    # Stage 1: Latent space coarse filter
    # Encode goal observation to latent
    goal_tokens = self.enc.forward(goal_obs)
    goal_z = self.dyn.encode_tokens(goal_tokens)

    candidates = []
    for traj_id, traj in enumerate(replay_buffer):
      for t in range(len(traj)):
        # Compute cosine distance in latent space
        traj_z = traj['latent'][t]
        dist = self._cosine_distance(traj_z, goal_z)
        if dist < 0.3:  # Coarse threshold
          candidates.append((traj_id, t, dist))

    # Take top 100 candidates
    candidates = sorted(candidates, key=lambda x: x[2])[:100]

    # Stage 2: Observation space precise matching
    try:
      from skimage.metrics import structural_similarity as ssim
    except ImportError:
      # Fallback: use latent space distance only
      return sorted(candidates, key=lambda x: x[2])[:top_k]

    results = []
    for traj_id, t, _ in candidates:
      # Decode latent to observation
      traj_latent = replay_buffer[traj_id]['latent'][t]
      obs_pred = self.dec.forward(traj_latent)

      # Compute SSIM
      similarity = ssim(obs_pred, goal_obs, multichannel=True, channel_axis=-1)
      if similarity > 0.7:  # Precise threshold
        results.append((traj_id, t, similarity))

    return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]

  def _cosine_distance(self, feat1, feat2):
    """Compute cosine distance between two feature vectors."""
    norm1 = jnp.linalg.norm(feat1) + 1e-8
    norm2 = jnp.linalg.norm(feat2) + 1e-8
    similarity = jnp.dot(feat1.flatten(), feat2.flatten()) / (norm1 * norm2)
    return 1.0 - similarity

  def imagine_to_goal(self, start_state, goal_obs):
    """Imagine trajectory from start_state to goal_obs (Case 2).

    Uses goal-conditioned policy and subgoal decomposition.

    Args:
      start_state: Current state (h, z)
      goal_obs: Goal observation

    Returns:
      List of trajectory dicts with {state, action, next_state}
    """
    if not self.fep_enabled:
      raise ValueError("FEP must be enabled to use imagine_to_goal")

    # Encode goal observation to latent
    goal_tokens = self.enc.forward(goal_obs)
    goal_state = self.dyn.encode_tokens(goal_tokens)

    # Decompose into subgoals
    subgoals = self.subgoal_decomposer.decompose(start_state, goal_state)

    # Imagine trajectory to each subgoal
    trajectory = []
    current_state = start_state

    for subgoal in subgoals:
      # Use goal-conditioned policy
      for _ in range(15):  # H steps per subgoal
        state_feat = self.feat2tensor(current_state)
        goal_feat = self.feat2tensor(subgoal)

        # Sample action from goal-conditioned policy
        action_dist = self.goal_cond_policy(state_feat, goal_feat)
        action = action_dist.sample(nj.seed())

        # Step world model
        next_state = self.dyn.step(current_state, action)

        trajectory.append({
            'state': current_state,
            'action': action,
            'next_state': next_state,
        })

        current_state = next_state

    return trajectory

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)


class GoalImaginator:
  """Maintains a goal state via EMA of high-reward features.

  Uses nj.Variable for persistent state (z_goal and reward_seen flag).
  Not included in optimizer modules since it has no trainable parameters.
  """

  def __init__(self, feat_dim, config, name='goal_imaginator'):
    self.feat_dim = feat_dim
    self.ema_rate = config.fep.goal_ema_rate
    self.topk = config.fep.goal_topk
    self._name = name

  def init_variables(self):
    """Initialize persistent state variables. Call after nj context is set."""
    self.z_goal = nj.Variable(
        jnp.zeros, self.feat_dim, f32, name=f'{self._name}_z_goal')
    self.reward_seen = nj.Variable(
        jnp.zeros, (), f32, name=f'{self._name}_reward_seen')

  def update_goal(self, feat, reward):
    """Update goal state using top-k high-reward features.

    Args:
      feat: Feature tensor [B, T, D] (already stop-gradiented)
      reward: Reward tensor [B, T]
    """
    B, T, D = feat.shape
    flat_feat = feat.reshape(B * T, D)
    flat_rew = reward.reshape(B * T)

    # Select top-k high reward states
    k = min(self.topk, B * T)
    topk_idx = jnp.argsort(flat_rew)[-k:]
    topk_feat = flat_feat[topk_idx]
    topk_rew = flat_rew[topk_idx]

    # Softmax-weighted average of top-k features
    weights = jax.nn.softmax(topk_rew)
    new_target = jnp.sum(topk_feat * weights[:, None], axis=0)

    # EMA update z_goal
    z_goal = self.z_goal.read()
    z_goal = (1 - self.ema_rate) * z_goal + self.ema_rate * new_target
    self.z_goal.write(z_goal)

    # Track whether any non-zero reward has been seen
    has_reward = (jnp.abs(flat_rew).max() > 1e-8).astype(f32)
    self.reward_seen.write(jnp.maximum(self.reward_seen.read(), has_reward))

  def get_goal(self):
    """Return current goal state."""
    return self.z_goal.read()

  def proximity(self, feat):
    """Compute cosine similarity between features and goal state.

    Args:
      feat: Feature tensor [..., D]

    Returns:
      Cosine similarity [...], range [-1, 1]
    """
    z_goal = self.z_goal.read()
    # Cosine similarity
    feat_norm = jnp.maximum(jnp.linalg.norm(feat, axis=-1, keepdims=True), 1e-8)
    goal_norm = jnp.maximum(jnp.linalg.norm(z_goal, keepdims=True), 1e-8)
    similarity = jnp.sum(feat * z_goal, axis=-1) / (
        feat_norm[..., 0] * goal_norm[0])
    # Mask out if reward never seen (cold start)
    return similarity * self.reward_seen.read()


class FEPScheduler:
  """Dynamic alpha/beta scheduler using z-score of prior entropy.

  Uses imagination-stage prior entropy as a forward-looking uncertainty
  signal. Computes z-score against running statistics to determine
  whether the agent should explore (high entropy) or exploit (low entropy).

  - z > 0: prior entropy above average → uncertain → increase alpha (explore)
  - z < 0: prior entropy below average → confident → increase beta (exploit)

  Uses nj.Variable for persistent state.
  """

  def __init__(self, config, name='fep_scheduler'):
    self.alpha_max = config.fep.alpha_max
    self.beta_max = config.fep.beta_max
    self.beta_warmup = config.fep.beta_warmup
    self.ema_rate = config.fep.scheduler_ema_rate
    self._name = name

  def init_variables(self):
    """Initialize persistent state variables. Call after nj context is set."""
    self.mean_ent = nj.Variable(
        jnp.zeros, (), f32, name=f'{self._name}_mean_ent')
    self.var_ent = nj.Variable(
        jnp.ones, (), f32, name=f'{self._name}_var_ent')
    self.step_count = nj.Variable(
        jnp.zeros, (), f32, name=f'{self._name}_step_count')

  def update_and_get(self, prior_entropy, training):
    """Update entropy stats and return current alpha, beta.

    Args:
      prior_entropy: Scalar, mean prior entropy from imagination stage
      training: Whether in training mode

    Returns:
      (alpha, beta, metrics) tuple where metrics is a dict
    """
    mean_ent = self.mean_ent.read()
    var_ent = self.var_ent.read()
    step_count = self.step_count.read()

    # EMA update of mean and variance
    new_mean = jnp.where(
        training,
        (1 - self.ema_rate) * mean_ent + self.ema_rate * prior_entropy,
        mean_ent)
    delta = prior_entropy - new_mean
    new_var = jnp.where(
        training,
        (1 - self.ema_rate) * var_ent + self.ema_rate * delta ** 2,
        var_ent)
    new_step = jnp.where(training, step_count + 1, step_count)

    self.mean_ent.write(new_mean)
    self.var_ent.write(new_var)
    self.step_count.write(new_step)

    # z-score: how unusual is current prior entropy
    std_ent = jnp.sqrt(new_var + 1e-8)
    z = (prior_entropy - new_mean) / std_ent

    # Alpha: z > 0 → entropy above average → explore
    alpha = self.alpha_max * jax.nn.sigmoid(3.0 * z)

    # Beta: z < 0 → entropy below average → exploit
    warmup_factor = jnp.clip(new_step / self.beta_warmup, 0.0, 1.0)
    beta = self.beta_max * warmup_factor * jax.nn.sigmoid(-3.0 * z)

    metrics = {
        'fep/prior_ent': prior_entropy,
        'fep/prior_ent_mean': new_mean,
        'fep/prior_ent_std': std_ent,
        'fep/z_score': z,
    }

    return alpha, beta, metrics

