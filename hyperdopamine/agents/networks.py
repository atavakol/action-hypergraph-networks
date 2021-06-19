from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin.tf

slim = tf.contrib.slim


gin.constant('networks.STACK_SIZE_1', 1)
gin.constant('networks.STACK_SIZE_4', 4)
gin.constant('networks.OBSERVATION_DTYPE_FLOAT32', tf.float32)


def _atari_dqn_network(num_actions, num_sub_actions, state, use_dueling):
  del num_sub_actions
  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  net = slim.conv2d(net, 32, [8, 8], stride=4)
  net = slim.conv2d(net, 64, [4, 4], stride=2)
  net = slim.conv2d(net, 64, [3, 3], stride=1)
  net = slim.flatten(net)
  if use_dueling:
    v_net = slim.fully_connected(net, 512)
    v_value = slim.fully_connected(v_net, 1, activation_fn=None)
    adv_net = slim.fully_connected(net, 512)
    adv_values = slim.fully_connected(adv_net, num_actions, 
                                      activation_fn=None)
    adv_values -= tf.reduce_mean(adv_values, axis=1, keepdims=True)
    q_values = adv_values + v_value
    return q_values, v_value
  else:
    net = slim.fully_connected(net, 512)
    q_values = slim.fully_connected(net, num_actions, 
                                    activation_fn=None)
    return q_values, None


@gin.configurable
def _atari_hgqn_network(num_actions, num_sub_actions, state, use_dueling, 
                        hyperedge_orders, mixer):
  assert num_sub_actions == [3,3,2]
  assert num_actions == np.prod(num_sub_actions)
  assert all(x in [1,2,3] for x in hyperedge_orders)
  ATARI_MIXER_NET_H = 25

  n = len(num_sub_actions)
  num_heads = int(0)
  if 1 in hyperedge_orders:
    num_heads += int(n)
  if 2 in hyperedge_orders:
    num_heads += int(n*(n-1)/2)
  if 3 in hyperedge_orders:
    num_heads += int(1)

  if num_heads == 3:
    assert (hyperedge_orders == [1]) or (hyperedge_orders == [2])
    LAYER_WIDTH = 172
  elif num_heads == 6:
    assert hyperedge_orders == [1,2]
    LAYER_WIDTH = 86
  elif num_heads == 4:
    assert (hyperedge_orders == [1,3]) or (hyperedge_orders == [2,3])
    LAYER_WIDTH = 128
  elif num_heads == 7:
    assert hyperedge_orders == [1,2,3]
    LAYER_WIDTH = 74
  else:
    assert num_heads == 1
    assert hyperedge_orders == [3]
    raise AssertionError('Use `atari_dqn_network` for [3].')
 
  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  net = slim.conv2d(net, 32, [8, 8], stride=4)
  net = slim.conv2d(net, 64, [4, 4], stride=2)
  net = slim.conv2d(net, 64, [3, 3], stride=1)
  net = slim.flatten(net)

  if list(filter((3).__ne__, hyperedge_orders)) == [] or mixer == 'sum':
    order_1_values = 0.0
    order_2_values = 0.0
    order_3_values = 0.0

  bq_values_orig = None
  p_bq_values_orig = None
  t_bq_values_orig = None

  if 1 in hyperedge_orders:
    bq_values = []
    for num_sub_actions_dim in num_sub_actions:
      bnet = slim.fully_connected(net, LAYER_WIDTH)
      bq_values_dim = slim.fully_connected(bnet, num_sub_actions_dim, 
                                           activation_fn=None)
      bq_values.append(bq_values_dim)
    bq_values_orig = bq_values
    if mixer == 'sum':
      from hyperdopamine.agents.utils import SUM_ORDER_1_MAP
      order_1_mapping_matrices = SUM_ORDER_1_MAP
      bq_values = tf.concat(bq_values, -1)
      order_1_mapping_matrices = tf.transpose(order_1_mapping_matrices)
      assert order_1_mapping_matrices.shape == (8, num_actions)
      order_1_values = tf.matmul(bq_values, order_1_mapping_matrices)
      assert order_1_values.shape == (state.shape[0], num_actions)

  if 2 in hyperedge_orders:
    p_bq_values = []
    branch_pairs = [[0, 1], [0, 2], [1, 2]]
    assert len(branch_pairs) == int(n*(n-1)/2)
    for pair_of_branches,test in zip(branch_pairs,[9,6,6]):
      branch_i, branch_j = pair_of_branches
      pair_output_length = \
          num_sub_actions[branch_i] * num_sub_actions[branch_j]
      assert pair_output_length == test
      p_bnet = slim.fully_connected(net, LAYER_WIDTH)
      p_bq_values_pair_of_dims = \
          slim.fully_connected(p_bnet, pair_output_length, 
                               activation_fn=None)
      p_bq_values.append(p_bq_values_pair_of_dims)
    p_bq_values_orig = p_bq_values
    if mixer == 'sum':
      from hyperdopamine.agents.utils import SUM_ORDER_2_MAP
      order_2_mapping_matrices = SUM_ORDER_2_MAP
      p_bq_values = tf.concat(p_bq_values, -1)
      order_2_mapping_matrices = tf.transpose(order_2_mapping_matrices)
      assert order_2_mapping_matrices.shape == (21, num_actions)
      order_2_values = tf.matmul(p_bq_values, order_2_mapping_matrices)
      assert order_2_values.shape == (state.shape[0], num_actions)

  if 3 in hyperedge_orders:
    t_bnet = slim.fully_connected(net, LAYER_WIDTH)
    order_3_values = slim.fully_connected(t_bnet, num_actions, 
                                          activation_fn=None)
    t_bq_values_orig = order_3_values
    assert order_3_values.shape == (state.shape[0], num_actions)

  if (list(filter((3).__ne__, hyperedge_orders)) != [] and 
      mixer == 'universal'):
    initializer_W = slim.initializers.xavier_initializer()
    initializer_b = slim.init_ops.zeros_initializer()
    W1 = tf.get_variable('W1', shape=[num_heads, ATARI_MIXER_NET_H], 
        initializer=initializer_W, dtype=tf.float32, trainable=True)
    b1 = tf.get_variable('b1', shape=[ATARI_MIXER_NET_H], 
        initializer=initializer_b, dtype=tf.float32, trainable=True)
    W2 = tf.get_variable('W2', shape=[ATARI_MIXER_NET_H, 1], 
        initializer=initializer_W, dtype=tf.float32, trainable=True)
    b2 = tf.get_variable('b2', shape=[1], 
        initializer=initializer_b, dtype=tf.float32, trainable=True)
    if 1 in hyperedge_orders:
      from hyperdopamine.agents.utils import GENERAL_ORDER_1_MAP
      order_1_mapping_matrices = GENERAL_ORDER_1_MAP
    if 2 in hyperedge_orders:
      from hyperdopamine.agents.utils import GENERAL_ORDER_2_MAP
      order_2_mapping_matrices = GENERAL_ORDER_2_MAP
    all_values_per_action = []
    for act in range(num_actions):
      values_to_mix_act = []
      if 1 in hyperedge_orders:
        l1_per_composite = []
        for bq,b2c in zip(bq_values, order_1_mapping_matrices[act]):
          b2c = tf.expand_dims(b2c, -1)
          out = tf.matmul(bq, b2c)
          out = tf.squeeze(out, -1)
          l1_per_composite.append(out)
        l1_per_composite = tf.stack(l1_per_composite, -1)
        values_to_mix_act.append(l1_per_composite)
      if 2 in hyperedge_orders:
        l2_per_composite = []
        for pbq,pb2c in zip(p_bq_values, order_2_mapping_matrices[act]):
          pb2c = tf.expand_dims(pb2c, -1)
          out = tf.matmul(pbq, pb2c)
          out = tf.squeeze(out, -1)
          l2_per_composite.append(out)
        l2_per_composite = tf.stack(l2_per_composite, -1)
        values_to_mix_act.append(l2_per_composite)
      input_values = tf.concat(values_to_mix_act, -1)
      assert input_values.shape == (state.shape[0], 
          num_heads - (1 if 3 in hyperedge_orders else 0))
      all_values_per_action.append(input_values)
    all_values_per_action = tf.stack(all_values_per_action, -1)
    if 3 in hyperedge_orders:
      all_values_per_action = tf.concat([all_values_per_action, \
          tf.reshape(order_3_values, [-1, 1, num_actions])], 1)
    assert all_values_per_action.shape == \
        (state.shape[0], num_heads, num_actions)
    all_values_per_action = tf.transpose(all_values_per_action, 
                                         perm=[2, 0, 1])
    all_values_per_action = tf.reshape(all_values_per_action, 
                                       [-1, num_heads])
    q_values = tf.add(tf.matmul(all_values_per_action, W1), b1)
    q_values = tf.nn.relu(q_values)
    q_values = tf.add(tf.matmul(q_values, W2), b2)
    q_values = tf.reshape(q_values, [num_actions, -1])
    q_values = tf.transpose(q_values)
  else:
    assert (list(filter((3).__ne__, hyperedge_orders)) == [] or 
            mixer == 'sum')
    q_values = order_1_values + order_2_values + order_3_values
  assert q_values.shape == (state.shape[0], num_actions)

  if use_dueling:
    v_net = slim.fully_connected(net, 512)
    v_value = slim.fully_connected(v_net, 1, activation_fn=None)
    q_values -= tf.reduce_mean(q_values, axis=1, keepdims=True)
    q_values += v_value
    return q_values, v_value, bq_values_orig, p_bq_values_orig, \
        t_bq_values_orig
  else:
    return q_values, None, bq_values_orig, p_bq_values_orig, \
        t_bq_values_orig


@gin.configurable
def _hgqn_network(num_actions, num_sub_actions, state, use_dueling, 
                  hyperedge_orders, mixer):
  n = len(num_sub_actions)
  m = max(num_sub_actions)
  assert max(num_sub_actions) == min(num_sub_actions)
  assert num_actions == m**n
  possible_factors = \
      list(filter((0).__ne__, range(-1,(n if n<4 else 4),1)))
  assert all(x in possible_factors for x in hyperedge_orders)

  MIXER_NET_H = 25
  SHARED_NET_H1 = 600
  SHARED_NET_H2 = 400
  LAST_NET_H = 400

  num_heads = int(0)
  if 1 in hyperedge_orders:
    num_heads += int(n)
  if 2 in hyperedge_orders:
    num_heads += int(n*(n-1)/2)
  if 3 in hyperedge_orders:
    num_heads += int(n*(n-1)*(n-2)/6)
  if -1 in hyperedge_orders:
    num_heads += int(1)
  from hyperdopamine.agents.utils import ceil_rounder
  LAYER_WIDTH = ceil_rounder(LAST_NET_H / num_heads)

  net = tf.cast(state, tf.float32)
  net = slim.flatten(net)
  if SHARED_NET_H1 is not None:
    net = slim.fully_connected(net, SHARED_NET_H1)
  if SHARED_NET_H2 is not None:
    net = slim.fully_connected(net, SHARED_NET_H2)

  if (list(filter((-1).__ne__, hyperedge_orders)) == [] or 
      mixer == 'sum'):
    order_1_values = 0.0
    order_2_values = 0.0
    order_3_values = 0.0
    order_Nv_values = 0.0

  bq_values_orig = None
  p_bq_values_orig = None
  t_bq_values_orig = None
  n_bq_values_orig = None

  if 1 in hyperedge_orders:
    bq_values = []
    for num_sub_actions_dim in num_sub_actions:
      bnet = slim.fully_connected(net, LAYER_WIDTH)
      bq_values_dim = slim.fully_connected(bnet, num_sub_actions_dim, 
                                           activation_fn=None)
      bq_values.append(bq_values_dim)
    bq_values_orig = bq_values
    if mixer == 'sum':
      from hyperdopamine.agents.utils import create_sum_order_1_map
      order_1_mapping_matrices = create_sum_order_1_map(
          num_branches=n, num_sub_actions_per_branch=m)
      bq_values = tf.concat(bq_values, -1)
      order_1_mapping_matrices = tf.transpose(order_1_mapping_matrices)
      assert order_1_mapping_matrices.shape == (n*m, num_actions)
      order_1_values = tf.matmul(bq_values, order_1_mapping_matrices)
      assert order_1_values.shape == (state.shape[0], num_actions)

  if 2 in hyperedge_orders:
    p_bq_values = []
    num_branch_pairs = int(n*(n-1)/2)
    for _ in range(num_branch_pairs):
      p_bnet = slim.fully_connected(net, LAYER_WIDTH)
      p_bq_values_pair_of_dims = \
          slim.fully_connected(p_bnet, m**2, activation_fn=None)
      p_bq_values.append(p_bq_values_pair_of_dims)
    p_bq_values_orig = p_bq_values
    if mixer == 'sum':
      from hyperdopamine.agents.utils import create_sum_order_2_map
      order_2_mapping_matrices = create_sum_order_2_map(
          num_branches=n, num_sub_actions_per_branch=m)
      p_bq_values = tf.concat(p_bq_values, -1)
      order_2_mapping_matrices = tf.transpose(order_2_mapping_matrices)
      assert order_2_mapping_matrices.shape == (num_branch_pairs*m**2, 
                                                num_actions)
      order_2_values = tf.matmul(p_bq_values, order_2_mapping_matrices)
      assert order_2_values.shape == (state.shape[0], num_actions)

  if 3 in hyperedge_orders:
    t_bq_values = []
    num_branch_triplets = int(n*(n-1)*(n-2)/6)
    for _ in range(num_branch_triplets):
      t_bnet = slim.fully_connected(net, LAYER_WIDTH)
      t_bq_values_triplet_of_dims = \
          slim.fully_connected(t_bnet, m**3, activation_fn=None)
      t_bq_values.append(t_bq_values_triplet_of_dims)
    t_bq_values_orig = t_bq_values
    if mixer == 'sum':
      from hyperdopamine.agents.utils import create_sum_order_3_map
      order_3_mapping_matrices = create_sum_order_3_map(
          num_branches=n, num_sub_actions_per_branch=m)
      t_bq_values = tf.concat(t_bq_values, -1)
      order_3_mapping_matrices = tf.transpose(order_3_mapping_matrices)
      assert order_3_mapping_matrices.shape == (num_branch_triplets*m**3, 
                                                num_actions)
      order_3_values = tf.matmul(t_bq_values, order_3_mapping_matrices)
      assert order_3_values.shape == (state.shape[0], num_actions)

  if -1 in hyperedge_orders:
    n_bnet = slim.fully_connected(net, LAYER_WIDTH)
    order_Nv_values = slim.fully_connected(n_bnet, num_actions, 
                                           activation_fn=None)
    n_bq_values_orig = order_Nv_values
    assert order_Nv_values.shape == (state.shape[0], num_actions)

  if (list(filter((-1).__ne__, hyperedge_orders)) != [] and 
      mixer == 'universal'):
    initializer_W = slim.initializers.xavier_initializer()
    initializer_b = slim.init_ops.zeros_initializer()
    W1 = tf.get_variable('W1', shape=[num_heads, MIXER_NET_H], 
        initializer=initializer_W, dtype=tf.float32, trainable=True)
    b1 = tf.get_variable('b1', shape=[MIXER_NET_H], 
        initializer=initializer_b, dtype=tf.float32, trainable=True)
    W2 = tf.get_variable('W2', shape=[MIXER_NET_H, 1], 
        initializer=initializer_W, dtype=tf.float32, trainable=True)
    b2 = tf.get_variable('b2', shape=[1], 
        initializer=initializer_b, dtype=tf.float32, trainable=True)

    all_values = []
    if 1 in hyperedge_orders:
      from hyperdopamine.agents.utils import create_general_order_1_map
      order_1_mapping_matrices = create_general_order_1_map(
          num_branches=n, num_sub_actions_per_branch=m)
      order_1_mapping_matrices = np.array(order_1_mapping_matrices)
      order_1_mapping_matrices = \
          np.transpose(order_1_mapping_matrices, (1, 2, 0))
      order_1_mapping_matrices = \
          np.expand_dims(order_1_mapping_matrices, axis=0)
      assert order_1_mapping_matrices.shape == (1, n, m, num_actions)
      bq_values = tf.concat(bq_values, -1)
      bq_values = tf.reshape(bq_values, [-1, n, m])
      bq_values = tf.expand_dims(bq_values, axis=-1)
      out_l1 = tf.multiply(bq_values, order_1_mapping_matrices)
      out_l1 = tf.reduce_sum(out_l1, axis=2)
      out_l1 = tf.transpose(out_l1, (0,2,1))
      assert out_l1.shape == (state.shape[0], num_actions, n)
      all_values.append(out_l1)
      del order_1_mapping_matrices
    if 2 in hyperedge_orders:
      from hyperdopamine.agents.utils import create_general_order_2_map
      order_2_mapping_matrices = create_general_order_2_map(
          num_branches=n, num_sub_actions_per_branch=m)
      order_2_mapping_matrices = np.array(order_2_mapping_matrices)
      order_2_mapping_matrices = \
          np.transpose(order_2_mapping_matrices, (1,2,0))
      order_2_mapping_matrices = \
          np.expand_dims(order_2_mapping_matrices, axis=0)
      assert order_2_mapping_matrices.shape == \
          (1, n*(n-1)/2, m**2, num_actions)
      p_bq_values = tf.concat(p_bq_values, -1)
      p_bq_values = tf.reshape(p_bq_values, 
                               [-1, int(n*(n-1)/2), int(m**2)])
      p_bq_values = tf.expand_dims(p_bq_values, axis=-1)
      out_l2 = tf.multiply(p_bq_values, order_2_mapping_matrices)
      out_l2 = tf.reduce_sum(out_l2, axis=2)
      out_l2 = tf.transpose(out_l2, (0,2,1))
      assert out_l2.shape == (state.shape[0], num_actions, n*(n-1)/2)
      all_values.append(out_l2)
      del order_2_mapping_matrices
    if 3 in hyperedge_orders:
      from hyperdopamine.agents.utils import create_general_order_3_map
      order_3_mapping_matrices = create_general_order_3_map(
          num_branches=n, num_sub_actions_per_branch=m)
      order_3_mapping_matrices = np.array(order_3_mapping_matrices)
      order_3_mapping_matrices = \
          np.transpose(order_3_mapping_matrices, (1,2,0))
      order_3_mapping_matrices = \
          np.expand_dims(order_3_mapping_matrices, axis=0)
      assert order_3_mapping_matrices.shape == \
          (1, n*(n-1)*(n-2)/6, m**3, num_actions)
      t_bq_values = tf.concat(t_bq_values, -1)
      t_bq_values = tf.reshape(t_bq_values, 
                               [-1, n*(n-1)*(n-2)/6, m**3])
      t_bq_values = tf.expand_dims(t_bq_values, axis=-1)
      out_l3 = tf.multiply(t_bq_values, order_3_mapping_matrices)
      out_l3 = tf.reduce_sum(out_l3, axis=2)
      out_l3 = tf.transpose(out_l3, (0,2,1))
      assert out_l3.shape == \
          (state.shape[0], num_actions, n*(n-1)*(n-2)/6)
      all_values.append(out_l3)
      del order_3_mapping_matrices
    if -1 in hyperedge_orders:
      out_ln = tf.expand_dims(order_Nv_values, axis=-1)
      all_values.append(out_ln)
    all_values_per_action = tf.concat(all_values, -1)
    all_values_per_action = \
        tf.transpose(all_values_per_action, (1,0,2))
    assert all_values_per_action.shape == \
        (num_actions, state.shape[0], num_heads)
    all_values_per_action = tf.reshape(all_values_per_action, 
                                       [-1, num_heads])
    q_values = tf.add(tf.matmul(all_values_per_action, W1), b1)
    q_values = tf.nn.relu(q_values)
    q_values = tf.add(tf.matmul(q_values, W2), b2)
    q_values = tf.reshape(q_values, [num_actions, -1])
    q_values = tf.transpose(q_values)
    del all_values_per_action
  else:
    assert (list(filter((-1).__ne__, hyperedge_orders)) == [] or 
            mixer == 'sum')
    q_values = order_1_values + order_2_values + order_3_values + \
        order_Nv_values
  assert q_values.shape == (state.shape[0], num_actions)

  if use_dueling:
    v_net = slim.fully_connected(net, LAST_NET_H)
    v_value = slim.fully_connected(v_net, 1, activation_fn=None)
    q_values -= tf.reduce_mean(q_values, axis=1, keepdims=True)
    q_values += v_value
    return q_values, v_value, bq_values_orig, p_bq_values_orig, \
        t_bq_values_orig, n_bq_values_orig
  else:
    return q_values, None, bq_values_orig, p_bq_values_orig, \
        t_bq_values_orig, n_bq_values_orig


@gin.configurable
def _branching_network(num_sub_actions, state, use_dueling):
  from hyperdopamine.agents.utils import ceil_rounder
  LAST_NET_H = 400
  LAYER_WIDTH = ceil_rounder(LAST_NET_H / len(num_sub_actions))
  net = tf.cast(state, tf.float32)
  net = slim.flatten(net)
  net = slim.fully_connected(net, 600) 
  net = slim.fully_connected(net, 400)
  bq_values = []
  for num_sub_actions_dim in num_sub_actions:
    bnet = slim.fully_connected(net, LAYER_WIDTH)
    bq_values_dim = \
        slim.fully_connected(bnet, num_sub_actions_dim, 
                             activation_fn=None)
    if use_dueling:
      bq_values_dim -= tf.reduce_mean(bq_values_dim, axis=1, 
                                      keepdims=True)
    bq_values.append(bq_values_dim)
  if use_dueling:
    v_net = slim.fully_connected(net, LAST_NET_H)
    v_value = slim.fully_connected(v_net, 1, activation_fn=None)  
    return bq_values, v_value
  else:
    return bq_values, None


@gin.configurable
def atari_dqn_network(num_actions, num_sub_actions, network_type, state, 
                      **kwargs):
  assert kwargs['hyperedge_orders'] == None and kwargs['mixer'] == None
  q_values, v_value = _atari_dqn_network(num_actions, num_sub_actions, 
                                         state, kwargs['use_dueling'])
  return network_type(q_values, v_value, None, None, None, None)


@gin.configurable
def atari_hgqn_network(num_actions, num_sub_actions, network_type, state, 
                       **kwargs):
  assert num_sub_actions == [3,3,2]
  if (kwargs['hyperedge_orders'] == None or 
      not kwargs['hyperedge_orders']):
    raise ValueError('Unspecified hyperedge orders.')
  assert (kwargs['mixer'] == 'sum' or kwargs['mixer'] == 'universal' or
          (kwargs['mixer'] == None and kwargs['hyperedge_orders'] == [3]))
  q_values, v_value, bq_values, pbq_values, tbq_values = \
      _atari_hgqn_network(num_actions, num_sub_actions, state, 
                          kwargs['use_dueling'], 
                          kwargs['hyperedge_orders'], 
                          kwargs['mixer'])
  return network_type(q_values, v_value, bq_values, pbq_values, 
                      tbq_values, None)


@gin.configurable
def hgqn_network(num_actions, num_sub_actions, network_type, state, 
                 **kwargs):
  assert num_sub_actions != int
  if (kwargs['hyperedge_orders'] == None or 
      not kwargs['hyperedge_orders']):
    raise ValueError('Unspecified hyperedge orders.')
  assert (kwargs['mixer'] == 'sum' or kwargs['mixer'] == 'universal' or
          (kwargs['mixer'] == None and kwargs['hyperedge_orders'] == [-1]))
  q_values, v_value, bq_values, pbq_values, tbq_values, nbq_values = \
      _hgqn_network(num_actions, num_sub_actions, state, 
                    kwargs['use_dueling'], 
                    kwargs['hyperedge_orders'], 
                    kwargs['mixer'])
  return network_type(q_values, v_value, bq_values, pbq_values, tbq_values, 
                      nbq_values)


@gin.configurable
def branching_network(num_sub_actions, network_type, state, **kwargs):
  assert num_sub_actions != int
  bq_values, v_value = _branching_network(num_sub_actions, state, 
                                          kwargs['use_dueling'])
  return network_type(bq_values, v_value)


@gin.configurable
def sum_mixer(network_type, utils, **kwargs):
  net = tf.stack(utils, axis=1)
  assert net.shape[1] == len(utils)
  q_value = tf.reduce_sum(utils, axis=0)
  q_value = tf.expand_dims(q_value, axis=1)
  if kwargs['use_dueling']:
    assert q_value.shape == kwargs['v_value'].shape
    q_value += kwargs['v_value']
  return network_type(q_value)


@gin.configurable
def monotonic_linear_mixer(network_type, utils, **kwargs):
  net = tf.stack(utils, axis=1)
  assert net.shape[1] == len(utils)
  W = tf.get_variable('W', shape=[len(utils), 1], dtype=tf.float32, 
      trainable=True, initializer=slim.initializers.xavier_initializer())
  b = tf.get_variable('b', shape=[1], dtype=tf.float32, trainable=True,
      initializer=slim.init_ops.zeros_initializer())
  W = tf.abs(W)
  q_value = tf.add(tf.matmul(net, W), b)
  if kwargs['use_dueling']:
    assert q_value.shape == kwargs['v_value'].shape
    q_value += kwargs['v_value']
  return network_type(q_value)


@gin.configurable
def monotonic_nonlinear_mixer(network_type, utils, **kwargs):
  net = tf.stack(utils, axis=1)
  assert net.shape[1] == len(utils)
  initializer_W = slim.initializers.xavier_initializer()
  initializer_b = slim.init_ops.zeros_initializer()
  MIXER_NET_H = 25
  W1 = tf.get_variable('W1', shape=[len(utils), MIXER_NET_H], 
      initializer=initializer_W, dtype=tf.float32, trainable=True)
  b1 = tf.get_variable('b1', shape=[MIXER_NET_H], 
      initializer=initializer_b, dtype=tf.float32, trainable=True)
  W2 = tf.get_variable('W2', shape=[MIXER_NET_H, 1], 
      initializer=initializer_W, dtype=tf.float32, trainable=True)
  b2 = tf.get_variable('b2', shape=[1], 
      initializer=initializer_b, dtype=tf.float32, trainable=True)
  W1 = tf.abs(W1)
  W2 = tf.abs(W2)
  net = tf.add(tf.matmul(net, W1), b1)
  net = tf.nn.elu(net)  # using ELU to maintain strict monotonicity
  q_value = tf.add(tf.matmul(net, W2), b2)
  if kwargs['use_dueling']:
    assert q_value.shape == kwargs['v_value'].shape
    q_value += kwargs['v_value']
  return network_type(q_value)
