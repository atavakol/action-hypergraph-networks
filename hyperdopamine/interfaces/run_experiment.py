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

import os
import sys
import time

from hyperdopamine.agents.dqn import dqn_agent
from hyperdopamine.agents.rainbow import rainbow_agent
from hyperdopamine.agents.bdqn import bdqn_agent
from hyperdopamine.agents.hgqn_r1 import hgqn_r1_agent
from hyperdopamine.interfaces import atari_lib
from hyperdopamine.interfaces import checkpointer
from hyperdopamine.interfaces import iteration_statistics
from hyperdopamine.interfaces import logger

import random 
import numpy as np
import tensorflow as tf
import gin.tf


def load_gin_configs(gin_files, gin_bindings):
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
        num_sub_actions=[sub_action_space.n for sub_action_space in \
            environment.sub_action_space],
        summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(sess, num_actions=environment.action_space.n,
        num_sub_actions=[sub_action_space.n for sub_action_space in \
            environment.sub_action_space],
        summary_writer=summary_writer)
  elif agent_name == 'hgqn_r1':
    return hgqn_r1_agent.HGQNr1Agent(
        sess, num_sub_actions=[sub_action_space.n for sub_action_space in \
            environment.action_space],
        summary_writer=summary_writer)
  elif agent_name == 'bdqn':
    return bdqn_agent.BDQNAgent(
        sess, num_sub_actions=[sub_action_space.n for sub_action_space in \
            environment.action_space],
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_runner(base_dir, schedule='continuous_train_and_eval'):
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


@gin.configurable
class Runner(object):
  ''' Object that handles running train-and-evaluate experiments. '''

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               agent_seed=None,
               checkpoint_file_prefix='ckpt',
               checkpoint_frequency=1,
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               render=False,
               reward_clipping=True):
    assert base_dir is not None
    self._checkpoint_frequency = checkpoint_frequency
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)
    self._render = render
    self._reward_clipping = reward_clipping
    self._environment = create_environment_fn()

    print('Seeding agent with seed', agent_seed)
    tf.set_random_seed(agent_seed)
    np.random.seed(agent_seed)
    random.seed(agent_seed)

    config = tf.ConfigProto(allow_soft_placement=True)
    self._sess = tf.Session('', config=config)
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)
    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())
    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

  def _create_directories(self):
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix,
                                                   self._checkpoint_frequency)
    self._start_iteration = 0
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        assert 'logs' in experiment_data
        assert 'current_iteration' in experiment_data
        self._logger.data = experiment_data['logs']
        self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def _initialize_episode(self):
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    observation, reward, is_terminal, info = self._environment.step(action)
    if self._render and self._agent.eval_mode:
      self._environment.render()
    return observation, reward, is_terminal

  def _end_episode(self, reward, is_timeout):
    self._agent.end_episode(reward, is_timeout)

  def _run_one_episode(self):
    step_number = 0
    total_reward = 0.
    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state or the time limit.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)
      total_reward += reward
      step_number += 1

      if self._reward_clipping:
        reward = np.clip(reward, -1, 1)

      if self._environment.game_over:
        # Stop the run loop once we reach the true end of episode.
        is_timeout = False
        break
      elif step_number == self._max_steps_per_episode:
        # Stop the run loop once we reach the timeout of episode.
        is_timeout = True
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent. Only meaningful for Atari domains,
        # otherwise will not reach here.
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward, is_timeout)
    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    step_count = 0
    num_episodes = 0
    sum_returns = 0.
    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    tf.logging.info('Average undiscounted return per training episode: %.2f',
                    average_return)
    tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    self._agent.eval_mode = True
    _, sum_returns, num_episodes = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics)
    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes',
                         simple_value=num_episodes_train),
        tf.Summary.Value(tag='Train/AverageReturns',
                         simple_value=average_reward_train),
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    # @TODO Render PyBullet only during evaluation.  
    if self._render and 'Bullet' in self._environment.spec.id:
        self._environment.render()

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      if iteration % self._checkpoint_frequency == 0:
        self._checkpoint_experiment(iteration)


@gin.configurable
class TrainRunner(Runner):
  ''' Object that handles running train-only experiments. '''

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    tf.logging.info('Creating TrainRunner...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)
    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)
