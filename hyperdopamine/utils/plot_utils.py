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

import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


FILE_PREFIX = 'log'
ITERATION_PREFIX = 'iteration_'

ALL_GAMES = ['AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
             'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk',
             'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede',
             'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
             'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
             'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
             'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
             'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix',
             'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid',
             'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris',
             'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham',
             'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge',
             'Zaxxon']


def load_baselines(base_dir, verbose=False):
  experimental_data = {}
  for game in ALL_GAMES:
    for agent in ['dqn', 'c51', 'rainbow', 'iqn']:
      game_data_file = os.path.join(base_dir, agent, '{}.pkl'.format(game))
      if not tf.gfile.Exists(game_data_file):
        if verbose:
          # pylint: disable=superfluous-parens
          print('Unable to load data for agent {} on game {}'.format(agent,
                                                                     game))
          # pylint: enable=superfluous-parens
        continue
      with tf.gfile.Open(game_data_file, 'rb') as f:
        if sys.version_info.major >= 3:
          # pylint: disable=unexpected-keyword-arg
          single_agent_data = pickle.load(f, encoding='latin1')
          # pylint: enable=unexpected-keyword-arg
        else:
          single_agent_data = pickle.load(f)
        single_agent_data['agent'] = agent
        if game in experimental_data:
          experimental_data[game] = experimental_data[game].merge(
              single_agent_data, how='outer')
        else:
          experimental_data[game] = single_agent_data
  return experimental_data


def load_statistics(log_path, iteration_number=None, verbose=True):
  if iteration_number is None:
    iteration_number = get_latest_iteration(log_path)
  log_file = '%s/%s_%d' % (log_path, FILE_PREFIX, iteration_number)
  if verbose:
    # pylint: disable=superfluous-parens
    print('Reading statistics from: {}'.format(log_file))
    # pylint: enable=superfluous-parens
  with tf.gfile.Open(log_file, 'rb') as f:
    return pickle.load(f), iteration_number


def get_latest_file(path):
  try:
    latest_iteration = get_latest_iteration(path)
    return os.path.join(path, '{}_{}'.format(FILE_PREFIX, latest_iteration))
  except ValueError:
    return None


def get_latest_iteration(path):
  glob = os.path.join(path, '{}_[0-9]*'.format(FILE_PREFIX))
  log_files = tf.gfile.Glob(glob)
  if not log_files:
    raise ValueError('No log data found at {}'.format(path))
  def extract_iteration(x):
    return int(x[x.rfind('_') + 1:])
  latest_iteration = max(extract_iteration(x) for x in log_files)
  return latest_iteration


def summarize_data(data, summary_keys):
  summary = {}
  latest_iteration_number = len(data.keys())
  current_value = None
  for key in summary_keys:
    summary[key] = []
    for i in range(latest_iteration_number):
      iter_key = '{}{}'.format(ITERATION_PREFIX, i)
      if iter_key in data:
        current_value = np.mean(data[iter_key][key])
      summary[key].append(current_value)
  return summary


def read_experiment(log_path,
                    parameter_set=None,
                    job_descriptor='',
                    iteration_number=None,
                    summary_keys=('train_episode_returns',
                                  'eval_episode_returns'),
                    verbose=False):
  keys = [] if parameter_set is None else list(parameter_set.keys())
  ordered_values = [parameter_set[key] for key in keys]
  column_names = keys + ['iteration'] + list(summary_keys)
  num_parameter_settings = len([_ for _ in itertools.product(*ordered_values)])
  expected_num_iterations = 200
  expected_num_rows = num_parameter_settings * expected_num_iterations
  data_frame = pd.DataFrame(index=np.arange(0, expected_num_rows),
                            columns=column_names)
  row_index = 0

  for parameter_tuple in itertools.product(*ordered_values):
    if job_descriptor is not None:
      name = job_descriptor.format(*parameter_tuple)
    else:
      name = '-'.join([keys[i] + '_' + str(parameter_tuple[i])
                       for i in range(len(keys))])

    experiment_path = '{}/{}/logs'.format(log_path, name)
    raw_data, last_iteration = load_statistics(
        experiment_path, iteration_number=iteration_number, verbose=verbose)

    summary = summarize_data(raw_data, summary_keys)
    for iteration in range(last_iteration):
      row_data = (list(parameter_tuple) + [iteration] +
                  [summary[key][iteration] for key in summary_keys])
      data_frame.loc[row_index] = row_data
      row_index += 1
  return data_frame.drop(np.arange(row_index, expected_num_rows))


def smooth_results(results, window_size, freq):
  from collections import deque
  y_queue = deque(maxlen=window_size)
  means = []
  stds = []
  results = np.array(results)
  seeds, timesteps = results.shape[:2]
  for t in range(timesteps):
    y_queue.append(results[:,t])
    if len(y_queue) < window_size or t % freq != 0: continue
    means.append(np.mean(y_queue))
    stds.append(np.std(y_queue))
  return None, np.array(means), np.array(stds)
