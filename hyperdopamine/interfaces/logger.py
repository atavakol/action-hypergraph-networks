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
import pickle

import tensorflow as tf


CHECKPOINT_DURATION = 4


class Logger(object):
  def __init__(self, logging_dir):
    self.data = {}
    self._logging_enabled = True

    if not logging_dir:
      tf.logging.info('Logging directory not specified, will not log.')
      self._logging_enabled = False
      return
    try:
      tf.gfile.MakeDirs(logging_dir)
    except tf.errors.PermissionDeniedError:
      pass
    if not tf.gfile.Exists(logging_dir):
      tf.logging.warning(
          'Could not create directory %s, logging will be disabled.',
          logging_dir)
      self._logging_enabled = False
      return
    self._logging_dir = logging_dir

  def __setitem__(self, key, value):
    if self._logging_enabled:
      self.data[key] = value

  def _generate_filename(self, filename_prefix, iteration_number):
    filename = '{}_{}'.format(filename_prefix, iteration_number)
    return os.path.join(self._logging_dir, filename)

  def log_to_file(self, filename_prefix, iteration_number):
    if not self._logging_enabled:
      tf.logging.warning('Logging is disabled.')
      return
    log_file = self._generate_filename(filename_prefix, iteration_number)
    with tf.gfile.GFile(log_file, 'w') as fout:
      pickle.dump(self.data, fout, protocol=pickle.HIGHEST_PROTOCOL)
    stale_iteration_number = iteration_number - CHECKPOINT_DURATION
    if stale_iteration_number >= 0:
      stale_file = self._generate_filename(filename_prefix,
                                           stale_iteration_number)
      try:
        tf.gfile.Remove(stale_file)
      except tf.errors.NotFoundError:
        pass

  def is_logging_enabled(self):
    return self._logging_enabled
