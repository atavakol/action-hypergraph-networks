# This script modifies and builds on Dopamine with the original copyright 
# note:
# 
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.spaces import Discrete
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf
import gin.tf
import cv2

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = (84, 84)
NATURE_DQN_DTYPE = tf.uint8
NATURE_DQN_STACK_SIZE = 4


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=True, 
                             environment_seed=None,
                             action_rep='gym_composite'):
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  print('Creating environment with name', full_game_name)
  env = gym.make(full_game_name)
  print('Seeding environment with seed', environment_seed)
  env.seed(environment_seed)
  env = env.env
  if action_rep == 'gym_composite':
    env = AtariGymActions(env)
  elif action_rep == 'cartesian_composite':
    env = AtariCartesianActions(env)
  elif action_rep == 'branching':
    env = AtariBranchingActions(env)
  else:
    raise NotImplementedError(action_rep)
  env = AtariPreprocessing(env)
  return env


class AtariGymActions(gym.ActionWrapper):
  ''' A wrapper class for adding subaction space metadata around an Atari 
  2600 environment. '''

  def __init__(self, env):
    super(AtariGymActions, self).__init__(env)
    assert env.action_space.n == 18, env.action_space.n
    self.action_space = env.action_space
    self.sub_action_space = [Discrete(3), Discrete(3), Discrete(2)]
    self.ale = env.ale
  
  def action(self, action):
    return action


class AtariCartesianActions(gym.ActionWrapper):
  ''' A wrapper class for forming a Cartesian action space around an Atari 
  2600 environment. '''

  def __init__(self, env):
    super(AtariCartesianActions, self).__init__(env)
    assert env.action_space.n == 18, env.action_space.n
    self.action_space = env.action_space
    self.sub_action_space = [Discrete(3), Discrete(3), Discrete(2)]
    self.action_map = np.empty(18, dtype=int)
    self.action_map[0]  = 0  # NOOP
    self.action_map[1]  = 1  # FIRE
    self.action_map[6]  = 2  # UP
    self.action_map[4]  = 3  # RIGHT
    self.action_map[2]  = 4  # LEFT
    self.action_map[12] = 5  # DOWN
    self.action_map[10] = 6  # UPRIGHT
    self.action_map[8]  = 7  # UPLEFT
    self.action_map[16] = 8  # DOWNRIGHT
    self.action_map[14] = 9  # DOWNLEFT
    self.action_map[7]  = 10 # UPFIRE
    self.action_map[5]  = 11 # RIGHTFIRE
    self.action_map[3]  = 12 # LEFTFIRE
    self.action_map[13] = 13 # DOWNFIRE
    self.action_map[11] = 14 # UPRIGHTFIRE
    self.action_map[9]  = 15 # UPLEFTFIRE
    self.action_map[17] = 16 # DOWNRIGHTFIRE
    self.action_map[15] = 17 # DOWNLEFTFIRE
    self.ale = env.ale
  
  def action(self, action):
    ale_action = self.action_map[action]
    return ale_action


class AtariBranchingActions(gym.ActionWrapper):
  ''' A wrapper class for branching the action space of an Atari 
  2600 environment. '''

  def __init__(self, env):
    super(AtariBranchingActions, self).__init__(env)
    assert env.action_space.n == 18, env.action_space.n
    self.action_space = [Discrete(3), Discrete(3), Discrete(2)]
    self.sub_action_space = self.action_space
    self.action_map = np.empty((3, 3, 2), dtype=int)
    self.action_map[0, 0, 0] = 0  # NOOP
    self.action_map[0, 0, 1] = 1  # FIRE
    self.action_map[1, 0, 0] = 2  # UP
    self.action_map[0, 2, 0] = 3  # RIGHT
    self.action_map[0, 1, 0] = 4  # LEFT
    self.action_map[2, 0, 0] = 5  # DOWN
    self.action_map[1, 2, 0] = 6  # UPRIGHT
    self.action_map[1, 1, 0] = 7  # UPLEFT
    self.action_map[2, 2, 0] = 8  # DOWNRIGHT
    self.action_map[2, 1, 0] = 9  # DOWNLEFT
    self.action_map[1, 0, 1] = 10 # UPFIRE
    self.action_map[0, 2, 1] = 11 # RIGHTFIRE
    self.action_map[0, 1, 1] = 12 # LEFTFIRE
    self.action_map[2, 0, 1] = 13 # DOWNFIRE
    self.action_map[1, 2, 1] = 14 # UPRIGHTFIRE
    self.action_map[1, 1, 1] = 15 # UPLEFTFIRE
    self.action_map[2, 2, 1] = 16 # DOWNRIGHTFIRE
    self.action_map[2, 1, 1] = 17 # DOWNLEFTFIRE
    self.ale = env.ale

  def action(self, action):
    up_down, left_right, fire = action
    ale_action = self.action_map[up_down, left_right, fire]
    return ale_action


class EnvSpec:
  def __init__(self, id):
    self.id = id


@gin.configurable
class AtariPreprocessing(object):
  ''' A class implementing image preprocessing for Atari 2600 agents. '''

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84):
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    self.screen_buffer = [
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0
    self.spec = EnvSpec(self.environment.spec.id)

  @property
  def observation_space(self):
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def sub_action_space(self):
    return self.environment.sub_action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def reset(self):
    self.environment.reset()
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode='human'):
    return self.environment.render(mode)

  def step(self, action):
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      if is_terminal:
        break
      elif time_step >= self.frame_skip - 2:
        t = time_step - (self.frame_skip - 2)
        self._fetch_grayscale_observation(self.screen_buffer[t])

    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                 out=self.screen_buffer[0])
    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
