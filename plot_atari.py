import os
import pickle

from hyperdopamine.utils import plot_utils
import numpy as np
import pandas as pd
import seaborn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['axes.linewidth'] = 1.
plt.rcParams.update({'font.size': 7})
plt.rcParams['font.family'] = 'Times New Roman, serif'


BASE_PATH = './data/benchmark/atari'
MAX_EPOCH = 200
COLORS = {'hgqn': '#000CF9', 'dqn': '#888888'}

TITLES = {
    'Alien': 'Alien',
    'BankHeist': 'Bank Heist', 
    'BattleZone': 'Battlezone', 
    'Berzerk': 'Berzerk',
    'Boxing': 'Boxing', 
    'Centipede': 'Centipede', 
    'ChopperCommand': 'Chopper Command',
    'DoubleDunk': 'Double Dunk', 
    'FishingDerby': 'Fishing Derby', 
    'Frostbite': 'Frostbite', 
    'Gravitar': 'Gravitar', 
    'Hero': 'H.E.R.O.', 
    'IceHockey': 'Ice Hockey', 
    'Jamesbond': 'James Bond 007',
    'Kangaroo': 'Kangaroo', 
    'Krull': 'Krull', 
    'MontezumaRevenge': 'Montezuma\'s Revenge', 
    'Pitfall': 'Pitfall!', 
    'PrivateEye': 'Private Eye', 
    'Riverraid': 'River Raid', 
    'RoadRunner': 'Road Runner',
    'Robotank': 'Robot Tank', 
    'Seaquest': 'Seaquest', 
    'Solaris': 'Solaris', 
    'StarGunner': 'Stargunner', 
    'Tennis': 'Tennis', 
    'Venture': 'Venture', 
    'YarsRevenge': 'Yars\' Revenge', 
    'Zaxxon': 'Zaxxon'}


def read_data(path, iteration_number):
  with open(path, 'rb') as f:
    d = pickle.load(f)
  seed_scores = []
  for i in range(iteration_number+1):
    episode_returns = d['iteration_'+str(i)]['train_episode_returns']
    seed_scores.append(np.mean(episode_returns))
  return np.array(seed_scores, dtype=np.float32)


def plot_performance(ax, game, agents):
  for agent in agents:
    df_pred = []
    df_time = []
    df_score = []
    log_path = BASE_PATH + '/{}/{}'.format(agent, game)
    SEEDS = os.listdir(log_path)
    for seed in SEEDS:
      df_pred.extend([agent]*MAX_EPOCH)
      df_time.extend(list(range(MAX_EPOCH)))
      seed_log_path = log_path + '/{}/logs'.format(seed)
      iteration_number = plot_utils.get_latest_iteration(seed_log_path)  
      seed_log_path = seed_log_path + '/log_' + str(iteration_number)
      df_score.extend(read_data(seed_log_path, iteration_number))
    df = pd.DataFrame(list(zip(df_pred, df_time, df_score)), 
                      columns=['predicate', 'Steps', 'Score'])
    seaborn.lineplot(x='Steps', y='Score', data=df, ci='sd', label=agent.upper(),
                     linewidth=1., color=COLORS[agent], ax=ax, legend=False)
    ax.lines[0].set_linestyle('--')
    ax.set_xlim(0,MAX_EPOCH)


GAMES = list(TITLES.keys())
GAMES = sorted(GAMES)

fig = plt.figure(figsize=(8, 10))
num_cols = 4
num_rows = len(GAMES) // num_cols + 1
for game_id, game in enumerate(GAMES):
  # print(game)
  ax = fig.add_subplot(num_rows, num_cols, game_id+1)
  plot_performance(ax, game, agents=['dqn','hgqn'])
  seaborn.despine(ax=ax)
  ax.set_title(TITLES[game], fontsize=9)
  ax.set_xlabel('Iteration', fontsize=7)
  ax.set_ylabel('Score', fontsize=7)

handles, labels = ax.get_legend_handles_labels()
axl = fig.add_subplot(num_rows, num_cols, game_id+2)
axl.axis('off')
axl.legend(handles, labels, loc='best', frameon=False, markerscale=2., fontsize=11)

plt.tight_layout()
fig.align_labels()
plt.savefig('atari_results.pdf')
print('figure saved')
