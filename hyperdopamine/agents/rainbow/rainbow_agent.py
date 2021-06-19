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

from hyperdopamine.agents.dqn import dqn_agent
from hyperdopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf
import gin.tf

slim = tf.contrib.slim


@gin.configurable
class RainbowAgent(dqn_agent.DQNAgent):
  ''' A simplified implementation of the Rainbow agent from
  Hessel et al. (2018) without C51 and noisy networks. '''

  def __init__(self,
               sess,
               num_actions,
               num_sub_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=dqn_agent.nature_dqn_network,
               hyperedge_orders=None,
               mixer=None,
               use_dueling=True,
               double_dqn=True,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=10000,
               update_period=1,
               target_update_period=2000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.05,
               epsilon_eval=0.001,
               epsilon_decay_period=50000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               loss_type='MSE',
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00001, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    tf.logging.info('\t double_dqn: %s', double_dqn)
    tf.logging.info('\t replay_scheme: %s', replay_scheme)

    self._double_dqn = double_dqn
    self._replay_scheme = replay_scheme
    self.optimizer = optimizer

    dqn_agent.DQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        num_sub_actions=num_sub_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        hyperedge_orders=hyperedge_orders,
        mixer=mixer,
        use_dueling=use_dueling,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        max_tf_checkpoints_to_keep=max_tf_checkpoints_to_keep,
        loss_type=loss_type,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _build_networks(self):
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)
    
    if self._double_dqn:
      self._replay_next_online_net_outputs = \
          self.online_convnet(self._replay.next_states)

  def _build_replay_buffer(self, use_staging):
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    if self._double_dqn:
      self._replay_next_online_net_q_argmax = tf.argmax(
          self._replay_next_online_net_outputs.q_values, axis=1)
      replay_next_online_net_q_argmax_one_hot = tf.one_hot(
          self._replay_next_online_net_q_argmax, self.num_actions, 1., 0., 
          name='replay_next_online_net_q_argmax_one_hot')
      replay_next_qt_max = tf.reduce_sum(
          self._replay_next_target_net_outputs.q_values * \
            replay_next_online_net_q_argmax_one_hot,
          reduction_indices=1,
          name='replay_next_qt_max')
    else:
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
    else:
      raise ValueError('Invalid loss type: {}'.format(self.loss_type))

    if self._replay_scheme == 'prioritized':
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar(self.loss_type+'Loss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        is_timeout=False,
                        priority=None):
    if priority is None:
      if is_timeout:
        priority = 0.
      elif self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority
    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)
