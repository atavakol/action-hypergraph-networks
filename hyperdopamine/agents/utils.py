import itertools
import math

import numpy as np


def ceil_rounder(x, base=5):
  return base * math.ceil(x / base)


def create_sum_order_1_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  sub_actions = []
  for _ in range(n):
    sub_actions.append(np.arange(0, m))
  composite_actions = list(itertools.product(*sub_actions))
  composite_actions = [list(comp_tup) for comp_tup in composite_actions]
  all_map_matrices_from_branches_to_composite = []
  for composite_a in composite_actions:
    a = np.array(composite_a)
    b = np.zeros((n, m))
    b[np.arange(a.size), a] = 1
    b = b.flatten()
    all_map_matrices_from_branches_to_composite.append(list(b))
  return all_map_matrices_from_branches_to_composite


def create_sum_order_2_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  num_branch_pairs = int(n*(n-1)/2)

  sub_actions = []
  for branch_id in range(n):
    sub_actions.append(np.arange(branch_id*m, (branch_id+1)*m))
  composite_actions = list(itertools.product(*sub_actions))

  branch_pairs = []
  all_edges = []
  for i in range(n):
    for j in range(i+1,n):
      branch_pairs.append([i,j])
      branch_pair_sub_actions = [sub_actions[i], sub_actions[j]]
      branch_pair_composite_sub_actions = \
          list(itertools.product(*branch_pair_sub_actions))
      all_edges.append(branch_pair_composite_sub_actions)
  assert len(branch_pairs) == len(all_edges)

  all_edge_indices_per_composite_action = []
  for composite_a in composite_actions:
    edge_id_per_branch_pair_for_composite_a = []
    for branch_pair_ids, branch_pair_composite_sub_actions in zip(
        reversed(branch_pairs), reversed(all_edges)):
      for edge_id, edge_sub_actions in enumerate(
          branch_pair_composite_sub_actions):
        if edge_sub_actions == (composite_a[branch_pair_ids[0]], 
                                composite_a[branch_pair_ids[1]]):
          edge_id_per_branch_pair_for_composite_a.append(edge_id)
          break
    edge_id_per_branch_pair_for_composite_a = \
        list(reversed(edge_id_per_branch_pair_for_composite_a))
    all_edge_indices_per_composite_action.append(
        edge_id_per_branch_pair_for_composite_a)
  assert len(all_edge_indices_per_composite_action) == \
      len(composite_actions) == m**n

  all_map_matrices_from_edges_to_composite = [] 
  for composite_a, edge_id_per_branch_pair_for_composite_a in zip(
      composite_actions, all_edge_indices_per_composite_action):
    a = np.array(edge_id_per_branch_pair_for_composite_a)
    b = np.zeros((num_branch_pairs, m**2))
    b[np.arange(a.size), a] = 1
    b = b.flatten()
    all_map_matrices_from_edges_to_composite.append(list(b))
  return all_map_matrices_from_edges_to_composite


def create_sum_order_3_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  num_branch_triplets = int(n*(n-1)*(n-2)/6)

  sub_actions = []
  for branch_id in range(n):
    sub_actions.append(np.arange(branch_id*m, (branch_id+1)*m))
  composite_actions = list(itertools.product(*sub_actions))

  branch_triplets = []
  all_triplets = []
  for i in range(n):
    for j in range(i+1,n):
      for k in range(j+1,n):
        branch_triplets.append([i,j,k])
        branch_triplet_sub_actions = [sub_actions[i], 
                                      sub_actions[j], 
                                      sub_actions[k]]
        branch_triplet_composite_sub_actions = \
          list(itertools.product(*branch_triplet_sub_actions))
        all_triplets.append(branch_triplet_composite_sub_actions)
  assert len(branch_triplets) == len(all_triplets)

  all_triplet_indices_per_composite_action = []
  for composite_a in composite_actions:
    edge_id_per_branch_triplet_for_composite_a = []
    for branch_triplet_ids, branch_triplet_composite_sub_actions in zip(
        reversed(branch_triplets), reversed(all_triplets)):
      for edge_id, edge_sub_actions in enumerate(
          branch_triplet_composite_sub_actions):
        if edge_sub_actions == (composite_a[branch_triplet_ids[0]], 
                                composite_a[branch_triplet_ids[1]],
                                composite_a[branch_triplet_ids[2]]):
          edge_id_per_branch_triplet_for_composite_a.append(edge_id)
          break
    edge_id_per_branch_triplet_for_composite_a = list(
        reversed(edge_id_per_branch_triplet_for_composite_a))
    all_triplet_indices_per_composite_action.append(
        edge_id_per_branch_triplet_for_composite_a)
  assert len(all_triplet_indices_per_composite_action) == \
      len(composite_actions) == m**n

  all_map_matrices_from_triplets_to_composite = [] 
  for composite_a, edge_id_per_branch_triplet_for_composite_a in zip(
      composite_actions, all_triplet_indices_per_composite_action):
    a = np.array(edge_id_per_branch_triplet_for_composite_a)
    b = np.zeros((num_branch_triplets, m**3))
    b[np.arange(a.size), a] = 1
    b = b.flatten()
    all_map_matrices_from_triplets_to_composite.append(list(b))
  return all_map_matrices_from_triplets_to_composite


def create_general_order_1_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  sub_actions = []
  for _ in range(n):
    sub_actions.append(np.arange(0, m))
  composite_actions = list(itertools.product(*sub_actions))
  composite_actions = [list(comp_tup) for comp_tup in composite_actions]
  all_map_matrices_from_branches_to_composite = []
  for composite_a in composite_actions:
    a = np.array(composite_a)
    b = np.zeros((n, m), np.float32)
    b[np.arange(a.size), a] = 1
    all_map_matrices_from_branches_to_composite.append(list(b))
  return all_map_matrices_from_branches_to_composite


def create_general_order_2_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  num_branch_pairs = int(n*(n-1)/2)

  sub_actions = []
  for branch_id in range(n):
    sub_actions.append(np.arange(branch_id*m, (branch_id+1)*m))
  composite_actions = list(itertools.product(*sub_actions))

  branch_pairs = []
  all_edges = []
  for i in range(n):
    for j in range(i+1,n):
      branch_pairs.append([i,j])
      branch_pair_sub_actions = [sub_actions[i], sub_actions[j]]
      branch_pair_composite_sub_actions = list(
          itertools.product(*branch_pair_sub_actions))
      all_edges.append(branch_pair_composite_sub_actions)
  assert len(branch_pairs) == len(all_edges)

  all_edge_indices_per_composite_action = []
  for composite_a in composite_actions:
    edge_id_per_branch_pair_for_composite_a = []
    for branch_pair_ids, branch_pair_composite_sub_actions in zip(
        reversed(branch_pairs), reversed(all_edges)):
      for edge_id, edge_sub_actions in enumerate(
          branch_pair_composite_sub_actions):
        if edge_sub_actions == (composite_a[branch_pair_ids[0]], 
                                composite_a[branch_pair_ids[1]]):
          edge_id_per_branch_pair_for_composite_a.append(edge_id)
          break
    edge_id_per_branch_pair_for_composite_a = list(
        reversed(edge_id_per_branch_pair_for_composite_a))
    all_edge_indices_per_composite_action.append(
        edge_id_per_branch_pair_for_composite_a)
  assert len(all_edge_indices_per_composite_action) == \
      len(composite_actions) == m**n

  all_map_matrices_from_edges_to_composite = [] 
  for composite_a, edge_id_per_branch_pair_for_composite_a in zip(
      composite_actions, all_edge_indices_per_composite_action):
    a = np.array(edge_id_per_branch_pair_for_composite_a)
    b = np.zeros((num_branch_pairs, m**2), np.float32)
    b[np.arange(a.size), a] = 1
    all_map_matrices_from_edges_to_composite.append(list(b))
  return all_map_matrices_from_edges_to_composite


def create_general_order_3_map(num_branches, num_sub_actions_per_branch):
  n = num_branches
  m = num_sub_actions_per_branch
  num_branch_triplets = int(n*(n-1)*(n-2)/6)

  sub_actions = []
  for branch_id in range(n):
    sub_actions.append(np.arange(branch_id*m, (branch_id+1)*m))
  composite_actions = list(itertools.product(*sub_actions))

  branch_triplets = []
  all_triplets = []
  for i in range(n):
    for j in range(i+1,n):
      for k in range(j+1,n):
        branch_triplets.append([i,j,k])
        branch_triplet_sub_actions = [sub_actions[i], 
                                      sub_actions[j], 
                                      sub_actions[k]]
        branch_triplet_composite_sub_actions = \
          list(itertools.product(*branch_triplet_sub_actions))
        all_triplets.append(branch_triplet_composite_sub_actions)
  assert len(branch_triplets) == len(all_triplets)

  all_triplet_indices_per_composite_action = []
  for composite_a in composite_actions:
    edge_id_per_branch_triplet_for_composite_a = []
    for branch_triplet_ids, branch_triplet_composite_sub_actions in zip(
        reversed(branch_triplets), reversed(all_triplets)):
      for edge_id, edge_sub_actions in enumerate(
          branch_triplet_composite_sub_actions):
        if edge_sub_actions == (composite_a[branch_triplet_ids[0]], 
                                composite_a[branch_triplet_ids[1]],
                                composite_a[branch_triplet_ids[2]]):
          edge_id_per_branch_triplet_for_composite_a.append(edge_id)
          break
    edge_id_per_branch_triplet_for_composite_a = list(
        reversed(edge_id_per_branch_triplet_for_composite_a))
    all_triplet_indices_per_composite_action.append(
        edge_id_per_branch_triplet_for_composite_a)
  assert len(all_triplet_indices_per_composite_action) == \
      len(composite_actions) == m**n

  all_map_matrices_from_triplets_to_composite = [] 
  for composite_a, edge_id_per_branch_triplet_for_composite_a in zip(
      composite_actions, all_triplet_indices_per_composite_action):
    a = np.array(edge_id_per_branch_triplet_for_composite_a)
    b = np.zeros((num_branch_triplets, m**3), np.float32)
    b[np.arange(a.size), a] = 1
    all_map_matrices_from_triplets_to_composite.append(list(b))
  return all_map_matrices_from_triplets_to_composite


SUM_ORDER_1_MAP = \
    [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]


SUM_ORDER_2_MAP = \
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


GENERAL_ORDER_1_MAP = \
    [[np.array([1., 0., 0.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([1., 0., 0.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([1., 0., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 1., 0.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([1., 0.], dtype=np.float32)],
     [np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 0., 1.], dtype=np.float32),
      np.array([0., 1.], dtype=np.float32)]]


GENERAL_ORDER_2_MAP = \
    [[np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 
      np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 
      np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)], 
     [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), 
      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)]]
