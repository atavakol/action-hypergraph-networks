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

import collections
import gzip
import math
import os
import pickle

import numpy as np
import tensorflow as tf
import gin.tf


ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))
STORE_FILENAME_PREFIX = '$store$_'
CHECKPOINT_DURATION = 4
MAX_SAMPLE_ATTEMPTS = 1000


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
  assert cursor < replay_capacity
  return np.array(
      [(cursor - update_horizon + i) % replay_capacity
       for i in range(stack_size + update_horizon)])


class OutOfGraphReplayBuffer(object):
  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    assert isinstance(observation_shape, tuple)
    if replay_capacity < update_horizon + stack_size:
      raise ValueError('There is not enough capacity to cover '
                       'update_horizon and stack_size.')

    tf.logging.info(
        'Creating a %s replay memory with the following parameters:',
        self.__class__.__name__)
    tf.logging.info('\t observation_shape: %s', str(observation_shape))
    tf.logging.info('\t observation_dtype: %s', str(observation_dtype))
    tf.logging.info('\t stack_size: %d', stack_size)
    tf.logging.info('\t replay_capacity: %d', replay_capacity)
    tf.logging.info('\t batch_size: %d', batch_size)
    tf.logging.info('\t update_horizon: %d', update_horizon)
    tf.logging.info('\t gamma: %f', gamma)

    self._action_shape = action_shape
    self._action_dtype = action_dtype
    self._reward_shape = reward_shape
    self._reward_dtype = reward_dtype
    self._observation_shape = observation_shape
    self._stack_size = stack_size
    self._state_shape = self._observation_shape + (self._stack_size,)
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._update_horizon = update_horizon
    self._gamma = gamma
    self._observation_dtype = observation_dtype
    self._max_sample_attempts = max_sample_attempts
    if extra_storage_types:
      self._extra_storage_types = extra_storage_types
    else:
      self._extra_storage_types = []
    self._create_storage()
    self.add_count = np.array(0)
    self.invalid_range = np.zeros((self._stack_size))
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon)],
        dtype=np.float32)

  def _create_storage(self):
    self._store = {}
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      self._store[storage_element.name] = np.empty(
          array_shape, dtype=storage_element.type)

  def get_add_args_signature(self):
    return self.get_storage_signature()

  def get_storage_signature(self):
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), np.uint8)
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def _add_zero_transition(self):
    zero_transition = []
    for element_type in self.get_add_args_signature():
      zero_transition.append(
          np.zeros(element_type.shape, dtype=element_type.type))
    self._add(*zero_transition)

  def add(self, observation, action, reward, terminal, *args):
    self._check_add_types(observation, action, reward, terminal, *args)
    if self.is_empty() or self._store['terminal'][self.cursor() - 1] == 1:
      for _ in range(self._stack_size - 1):
        self._add_zero_transition()
    self._add(observation, action, reward, terminal, *args)

  def _add(self, *args):
    cursor = self.cursor()

    arg_names = [e.name for e in self.get_add_args_signature()]
    for arg_name, arg in zip(arg_names, args):
      self._store[arg_name][cursor] = arg

    self.add_count += 1
    self.invalid_range = invalid_range(
        self.cursor(), self._replay_capacity, self._stack_size,
        self._update_horizon)

  def _check_add_types(self, *args):
    if len(args) != len(self.get_add_args_signature()):
      raise ValueError('Add expects {} elements, received {}'.format(
          len(self.get_add_args_signature()), len(args)))
    for arg_element, store_element in zip(args, self.get_add_args_signature()):
      if isinstance(arg_element, np.ndarray):
        arg_shape = arg_element.shape
      elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
        arg_shape = np.array(arg_element).shape
      else:
        arg_shape = tuple()
      store_element_shape = tuple(store_element.shape)
      if arg_shape != store_element_shape:
        raise ValueError('arg has shape {}, expected {}'.format(
            arg_shape, store_element_shape))

  def is_empty(self):
    return self.add_count == 0

  def is_full(self):
    return self.add_count >= self._replay_capacity

  def cursor(self):
    return self.add_count % self._replay_capacity

  def get_range(self, array, start_index, end_index):
    assert end_index > start_index, 'end_index must be larger than start_index'
    assert end_index >= 0
    assert start_index < self._replay_capacity
    if not self.is_full():
      assert end_index <= self.cursor(), (
          'Index {} has not been added.'.format(start_index))

    if start_index % self._replay_capacity < end_index % self._replay_capacity:
      return_array = array[start_index:end_index, ...]
    else:
      indices = [(start_index + i) % self._replay_capacity
                 for i in range(end_index - start_index)]
      return_array = array[indices, ...]
    return return_array

  def get_observation_stack(self, index):
    return self._get_element_stack(index, 'observation')

  def _get_element_stack(self, index, element_name):
    state = self.get_range(self._store[element_name],
                           index - self._stack_size + 1, index + 1)
    return np.moveaxis(state, 0, -1)

  def get_terminal_stack(self, index):
    return self.get_range(self._store['terminal'], index - self._stack_size + 1,
                          index + 1)

  def is_valid_transition(self, index):
    if index < 0 or index >= self._replay_capacity:
      return False
    if not self.is_full():
      if index >= self.cursor() - self._update_horizon:
        return False
      if index < self._stack_size - 1:
        return False
    if index in set(self.invalid_range):
      return False
    if self.get_terminal_stack(index)[:-1].any():
      return False
    return True

  def _create_batch_arrays(self, batch_size):
    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = []
    for element in transition_elements:
      batch_arrays.append(np.empty(element.shape, dtype=element.type))
    return tuple(batch_arrays)

  def sample_index_batch(self, batch_size):
    if self.is_full():
      min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
    else:
      min_id = self._stack_size - 1
      max_id = self.cursor() - self._update_horizon
      if max_id <= min_id:
        raise RuntimeError('Cannot sample a batch with fewer than stack size '
                           '({}) + update_horizon ({}) transitions.'.
                           format(self._stack_size, self._update_horizon))

    indices = []
    attempt_count = 0
    while (len(indices) < batch_size and
           attempt_count < self._max_sample_attempts):
      attempt_count += 1
      index = np.random.randint(min_id, max_id) % self._replay_capacity
      if self.is_valid_transition(index):
        indices.append(index)
    if len(indices) != batch_size:
      raise RuntimeError(
          'Max sample attempts: Tried {} times but only sampled {}'
          ' valid indices. Batch size is {}'.
          format(self._max_sample_attempts, len(indices), batch_size))

    return indices

  def sample_transition_batch(self, batch_size=None, indices=None):
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)
    for batch_element, state_index in enumerate(indices):
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                      0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (
          self._cumulative_discount_vector[:trajectory_length])
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)

      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          element_array[batch_element] = np.sum(
              trajectory_discount_vector * trajectory_rewards, axis=0)
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name in ('next_action', 'next_reward'):
          element_array[batch_element] = (
              self._store[element.name.lstrip('next_')][(next_state_index) %
                                                        self._replay_capacity])
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name in self._store.keys():
          element_array[batch_element] = (
              self._store[element.name][state_index])

    return batch_arrays

  def get_transition_elements(self, batch_size=None):
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        ReplayElement('state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('next_state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('next_action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('terminal', (batch_size,), np.uint8),
        ReplayElement('indices', (batch_size,), np.int32)
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                        element.type))
    return transition_elements

  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

  def _return_checkpointable_elements(self):
    checkpointable_elements = {}
    for member_name, member in self.__dict__.items():
      if member_name == '_store':
        for array_name, array in self._store.items():
          checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
      elif not member_name.startswith('_'):
        checkpointable_elements[member_name] = member
    return checkpointable_elements

  def save(self, checkpoint_dir, iteration_number):
    if not tf.gfile.Exists(checkpoint_dir):
      return

    checkpointable_elements = self._return_checkpointable_elements()

    for attr in checkpointable_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      with tf.gfile.Open(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f) as outfile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)

      stale_iteration_number = iteration_number - CHECKPOINT_DURATION
      if stale_iteration_number >= 0:
        stale_filename = self._generate_filename(checkpoint_dir, attr,
                                                 stale_iteration_number)
        try:
          tf.gfile.Remove(stale_filename)
        except tf.errors.NotFoundError:
          pass

  def load(self, checkpoint_dir, suffix):
    save_elements = self._return_checkpointable_elements()
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      if not tf.gfile.Exists(filename):
        raise tf.errors.NotFoundError(None, None,
                                      'Missing file: {}'.format(filename))
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      with tf.gfile.Open(filename, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as infile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            self._store[array_name] = np.load(infile, allow_pickle=False)
          elif isinstance(self.__dict__[attr], np.ndarray):
            self.__dict__[attr] = np.load(infile, allow_pickle=False)
          else:
            self.__dict__[attr] = pickle.load(infile)


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedReplayBuffer(object):
  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    if replay_capacity < update_horizon + 1:
      raise ValueError(
          'Update horizon ({}) should be significantly smaller '
          'than replay capacity ({}).'.format(update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = OutOfGraphReplayBuffer(
          observation_shape,
          stack_size,
          replay_capacity,
          batch_size,
          update_horizon,
          gamma,
          max_sample_attempts,
          observation_dtype=observation_dtype,
          extra_storage_types=extra_storage_types,
          action_shape=action_shape,
          action_dtype=action_dtype,
          reward_shape=reward_shape,
          reward_dtype=reward_dtype)

    self.create_sampling_ops(use_staging)

  def add(self, observation, action, reward, terminal, *args):
    self.memory.add(observation, action, reward, terminal, *args)

  def create_sampling_ops(self, use_staging):
    with tf.name_scope('sample_replay'):
      with tf.device('/cpu:*'):
        transition_type = self.memory.get_transition_elements()
        transition_tensors = tf.py_func(
            self.memory.sample_transition_batch, [],
            [return_entry.type for return_entry in transition_type],
            name='replay_sample_py_func')
        self._set_transition_shape(transition_tensors, transition_type)
        if use_staging:
          transition_tensors = self._set_up_staging(transition_tensors)
          self._set_transition_shape(transition_tensors, transition_type)
        self.unpack_transition(transition_tensors, transition_type)

  def _set_transition_shape(self, transition, transition_type):
    for element, element_type in zip(transition, transition_type):
      element.set_shape(element_type.shape)

  def _set_up_staging(self, transition):
    transition_type = self.memory.get_transition_elements()

    prefetch_area = tf.contrib.staging.StagingArea(
        [shape_with_type.type for shape_with_type in transition_type])

    self._prefetch_batch = prefetch_area.put(transition)
    initial_prefetch = tf.cond(
        tf.equal(prefetch_area.size(), 0),
        lambda: prefetch_area.put(transition), tf.no_op)

    with tf.control_dependencies([self._prefetch_batch, initial_prefetch]):
      prefetched_transition = prefetch_area.get()

    return prefetched_transition

  def unpack_transition(self, transition_tensors, transition_type):
    self.transition = collections.OrderedDict()
    for element, element_type in zip(transition_tensors, transition_type):
      self.transition[element_type.name] = element

    self.states = self.transition['state']
    self.actions = self.transition['action']
    self.rewards = self.transition['reward']
    self.next_states = self.transition['next_state']
    self.next_actions = self.transition['next_action']
    self.next_rewards = self.transition['next_reward']
    self.terminals = self.transition['terminal']
    self.indices = self.transition['indices']

  def save(self, checkpoint_dir, iteration_number):
    self.memory.save(checkpoint_dir, iteration_number)

  def load(self, checkpoint_dir, suffix):
    self.memory.load(checkpoint_dir, suffix)
