from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import gym
from gym.spaces import Discrete
from hyperdopamine.interfaces.gym_lib import GymPreprocessing
import numpy as np
import gin.tf


@gin.configurable
def create_discretised_environment(environment_name=None, version='v0', 
                                   action_rep='composite', compose_config=None,
                                   num_sub_actions=5, environment_seed=None):
  assert environment_name is not None
  if '_' in environment_name:  # DeepMind Control Suite.
    import hyperdopamine.interfaces.dmc_lib as dmc
    print('Creating environment with name', environment_name)
    print('Seeding environment with seed', environment_seed)
    env = dmc.make(environment_name, seed=environment_seed)
  else:  # OpenAI Gym.
    if 'Env' in environment_name:
      import pybullet_envs  # PyBullet.
    full_game_name = '{}-{}'.format(environment_name, version)
    print('Creating environment with name', full_game_name)
    env = gym.make(full_game_name)
    print('Seeding environment with seed', environment_seed)
    env.seed(environment_seed)
    env = env.env
  
  if action_rep == 'composite':
    env = ContinuousActionComposite(env, num_sub_actions=num_sub_actions)
  elif action_rep == 'branching':
    env = ContinuousActionBranching(env, num_sub_actions=num_sub_actions)
  elif action_rep == 'hybrid':
    env = ContinuousActionHybrid(env, num_sub_actions=num_sub_actions, 
                                 compose_config=compose_config)
  else: 
    raise NotImplementedError(action_rep)
  env = GymPreprocessing(env)
  return env


@gin.configurable
class ContinuousActionComposite(gym.ActionWrapper):
  def __init__(self, env, num_sub_actions): 
    super(ContinuousActionComposite, self).__init__(env)
    if type(num_sub_actions) == int:
      num_sub_actions = [num_sub_actions for _ in range(env.action_space.shape[0])]
    self.sub_actions = [np.linspace(min_a, max_a, n) for min_a, max_a, n in 
        zip(env.action_space.low, env.action_space.high, num_sub_actions)]
    self.cartesian_prod_actions = list(itertools.product(*self.sub_actions))
    self.action_space = Discrete(len(self.cartesian_prod_actions))
    self.sub_action_space = [Discrete(n) for n in num_sub_actions]

  def action(self, action):
    return np.array(self.cartesian_prod_actions[action])


@gin.configurable
class ContinuousActionBranching(gym.ActionWrapper):
  def __init__(self, env, num_sub_actions): 
    super(ContinuousActionBranching, self).__init__(env)
    if type(num_sub_actions) == int:
      num_sub_actions = [num_sub_actions for _ in range(env.action_space.shape[0])]
    self.sub_actions = [np.linspace(min_a, max_a, n) for min_a, max_a, n in 
        zip(env.action_space.low, env.action_space.high, num_sub_actions)]
    self.action_space = [Discrete(n) for n in num_sub_actions]
    self.sub_action_space = self.action_space

  def action(self, action):
    return np.array([s[a] for s, a in zip(self.sub_actions, action)])


@gin.configurable
class ContinuousActionHybrid(gym.ActionWrapper):
  def __init__(self, env, num_sub_actions, compose_config):
    super(ContinuousActionHybrid, self).__init__(env)
    if type(num_sub_actions) == int:
      num_sub_actions = [num_sub_actions for _ in range(env.action_space.shape[0])]
    self.sub_actions = [np.linspace(min_a, max_a, n) for min_a, max_a, n in 
        zip(env.action_space.low, env.action_space.high, num_sub_actions)]
    self.compose_config = compose_config

    compose_config_flat = \
      [dim_idx for compose_idxs in self.compose_config for dim_idx in compose_idxs]
    assert len(compose_config_flat) == len(set(compose_config_flat))
    assert set(np.arange(env.action_space.shape[0])) == set(compose_config_flat)

    self.hybrid_sub_actions = []
    for compose_idxs in self.compose_config:
      sub_actions_to_compose = [self.sub_actions[dim_idx] for dim_idx in compose_idxs]
      self.hybrid_sub_actions.append(list(itertools.product(*sub_actions_to_compose)))

    self.action_space = [Discrete(len(ha)) for ha in self.hybrid_sub_actions]
    self.sub_action_space = [Discrete(n) for n in num_sub_actions]
  
  def action(self, action):
    ctrls_unordered = np.array([hs[a] for hs,a in zip(self.hybrid_sub_actions, action)])
    dim_ctrls_unordered = \
      [dim_ctrl for head_ctrl in ctrls_unordered for dim_ctrl in head_ctrl]
    ctrls_ordered = [None for _ in dim_ctrls_unordered]

    i = 0
    for compose_idxs in self.compose_config:
      for dim_idx in compose_idxs:
        ctrls_ordered[dim_idx] = dim_ctrls_unordered[i]
        i += 1
    return ctrls_ordered
