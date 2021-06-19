# This script modifies Dopamine with the original copyright note:
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

from hyperdopamine.replay_memory import circular_replay_buffer
from hyperdopamine.replay_memory import sum_tree
from hyperdopamine.replay_memory.circular_replay_buffer import ReplayElement
import numpy as np
import tensorflow as tf
import gin.tf


class OutOfGraphPrioritizedReplayBuffer(
    circular_replay_buffer.OutOfGraphReplayBuffer):
  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=circular_replay_buffer.MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    super(OutOfGraphPrioritizedReplayBuffer, self).__init__(
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

    self.sum_tree = sum_tree.SumTree(replay_capacity)

  def get_add_args_signature(self):
    parent_add_signature = super(OutOfGraphPrioritizedReplayBuffer,
                                 self).get_add_args_signature()
    add_signature = parent_add_signature + [
        ReplayElement('priority', (), np.float32)
    ]
    return add_signature

  def _add(self, *args):
    parent_add_args = []
    for i, element in enumerate(self.get_add_args_signature()):
      if element.name == 'priority':
        priority = args[i]
      else:
        parent_add_args.append(args[i])
    self.sum_tree.set(self.cursor(), priority)
    super(OutOfGraphPrioritizedReplayBuffer, self)._add(*parent_add_args)

  def sample_index_batch(self, batch_size):
    indices = self.sum_tree.stratified_sample(batch_size)
    allowed_attempts = self._max_sample_attempts
    for i in range(len(indices)):
      if not self.is_valid_transition(indices[i]):
        if allowed_attempts == 0:
          raise RuntimeError(
              'Max sample attempts: Tried {} times but only sampled {}'
              ' valid indices. Batch size is {}'.
              format(self._max_sample_attempts, i, batch_size))
        index = indices[i]
        while not self.is_valid_transition(index) and allowed_attempts > 0:
          index = self.sum_tree.sample()
          allowed_attempts -= 1
        indices[i] = index
    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    transition = (super(OutOfGraphPrioritizedReplayBuffer, self).
                  sample_transition_batch(batch_size, indices))
    transition_elements = self.get_transition_elements(batch_size)
    transition_names = [e.name for e in transition_elements]
    probabilities_index = transition_names.index('sampling_probabilities')
    indices_index = transition_names.index('indices')
    indices = transition[indices_index]
    transition[probabilities_index][:] = self.get_priority(indices)
    return transition

  def set_priority(self, indices, priorities):
    assert indices.dtype == np.int32, ('Indices must be integers, '
                                       'given: {}'.format(indices.dtype))
    for index, priority in zip(indices, priorities):
      self.sum_tree.set(index, priority)

  def get_priority(self, indices):
    assert indices.shape, 'Indices must be an array.'
    assert indices.dtype == np.int32, ('Indices must be int32s, '
                                       'given: {}'.format(indices.dtype))
    batch_size = len(indices)
    priority_batch = np.empty((batch_size), dtype=np.float32)
    for i, memory_index in enumerate(indices):
      priority_batch[i] = self.sum_tree.get(memory_index)
    return priority_batch

  def get_transition_elements(self, batch_size=None):
    parent_transition_type = (
        super(OutOfGraphPrioritizedReplayBuffer,
              self).get_transition_elements(batch_size))
    probablilities_type = [
        ReplayElement('sampling_probabilities', (batch_size,), np.float32)
    ]
    return parent_transition_type + probablilities_type


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedPrioritizedReplayBuffer(
    circular_replay_buffer.WrappedReplayBuffer):
  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=circular_replay_buffer.MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    memory = OutOfGraphPrioritizedReplayBuffer(
        observation_shape, stack_size, replay_capacity, batch_size,
        update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)
    super(WrappedPrioritizedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        wrapped_memory=memory,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

  def tf_set_priority(self, indices, priorities):
    return tf.py_func(
        self.memory.set_priority, [indices, priorities], [],
        name='prioritized_replay_set_priority_py_func')

  def tf_get_priority(self, indices):
    return tf.py_func(
        self.memory.get_priority, [indices],
        tf.float32,
        name='prioritized_replay_get_priority_py_func')
