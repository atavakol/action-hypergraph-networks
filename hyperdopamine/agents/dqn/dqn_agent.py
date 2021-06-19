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

import collections
import math
import os
import random

from hyperdopamine.interfaces import atari_lib
from hyperdopamine.agents import networks
from hyperdopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf
import gin.tf

slim = tf.contrib.slim


NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE
nature_dqn_network = networks.atari_dqn_network


@gin.configurable
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


@gin.configurable
class DQNAgent(object):
  ''' An implementation of the DQN agent from Mnih et al. (2015). '''

  def __init__(self,
               sess,
               num_actions,
               num_sub_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=networks.atari_dqn_network,
               hyperedge_orders=None,
               mixer=None,
               use_dueling=False,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               loss_type='Huber',
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500):
    assert isinstance(observation_shape, tuple)
    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t num_actions: %d', num_actions)
    tf.logging.info('\t num_sub_actions: %s', str(num_sub_actions))
    tf.logging.info('\t network: %s', str(network))
    tf.logging.info('\t hyperedge_orders: %s', str(hyperedge_orders))
    tf.logging.info('\t mixer: %s', str(mixer))
    tf.logging.info('\t use_dueling: %s', use_dueling)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t update_horizon: %f', update_horizon)
    tf.logging.info('\t min_replay_history: %d', min_replay_history)
    tf.logging.info('\t update_period: %d', update_period)
    tf.logging.info('\t target_update_period: %d', target_update_period)
    tf.logging.info('\t epsilon_train: %f', epsilon_train)
    tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
    tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t loss_type: %s', loss_type)
    tf.logging.info('\t optimizer: %s', optimizer)

    self.num_actions = num_actions
    self.num_sub_actions = num_sub_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.hyperedge_orders = eval(hyperedge_orders) if \
        type(hyperedge_orders) == str else hyperedge_orders
    self.mixer = mixer
    self.use_dueling = use_dueling
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = False
    self.training_steps = 0
    self.loss_type = loss_type
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency

    with tf.device(tf_device):
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                     name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)
      self._build_networks()
      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)
    self._observation = None
    self._last_observation = None

  def _get_network_type(self):
    return collections.namedtuple('DQN_network', ['q_values',
        'v_value', 'bq_values', 'pbq_values', 'tbq_values', 'nbq_values'])

  def _network_template(self, state):
    kwargs = {}
    kwargs['use_dueling'] = self.use_dueling
    kwargs['hyperedge_orders'] = self.hyperedge_orders
    kwargs['mixer'] = self.mixer
    return self.network(self.num_actions, self.num_sub_actions,
        self._get_network_type(), state, **kwargs)

  def _build_networks(self):
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

  def _build_replay_buffer(self, use_staging):
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    if self.loss_type == 'Huber':
      loss = tf.losses.huber_loss(
          target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    elif self.loss_type == 'MSE':
      loss = tf.losses.mean_squared_error(
          target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar(self.loss_type+'Loss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def _build_sync_op(self):
    sync_qt_ops = []
    trainables_online = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
    trainables_target = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
    for (w_online, w_target) in zip(trainables_online, trainables_target):
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops

  def begin_episode(self, observation):
    self._reset_state()
    self._record_observation(observation)
    if not self.eval_mode:
      self._train_step()
    self.action = self._select_action()
    return self.action

  def step(self, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)
    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, 
                             False)
      self._train_step()
    self.action = self._select_action()
    return self.action

  def end_episode(self, reward, is_timeout=False):
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward, True, 
                             is_timeout)

  def _select_action(self):
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      return self._sess.run(self._q_argmax, {self.state_ph: self.state})

  def _train_step(self):
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)
    self.training_steps += 1

  def _record_observation(self, observation):
    self._observation = np.reshape(observation, self.observation_shape)
    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation

  def _store_transition(self, last_observation, action, reward, is_terminal, 
                        is_timeout=False):
    self._replay.add(last_observation, action, reward, is_terminal)

  def _reset_state(self):
    self.state.fill(0)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['eval_mode'] = self.eval_mode
    bundle_dictionary['training_steps'] = self.training_steps
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    try:
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      return False
    for key in self.__dict__:
      if key in bundle_dictionary:
        self.__dict__[key] = bundle_dictionary[key]
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
