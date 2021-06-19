from hyperdopamine.utils import plot_utils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

mpl.rcParams['axes.linewidth'] = 1.
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Times New Roman, serif'


BASE_PATH = './data/benchmark'
GAMES = [
    'ReacherBulletEnv', 
    'HopperBulletEnv', 
    'HalfCheetahBulletEnv', 
    'Walker2DBulletEnv', 
    'AntBulletEnv', 
    'Humanoid']
SEEDS = [str(i) for i in range(0,9)]
MAX_EPOCH = 300


TITLES = {
    'ReacherBulletEnv': r'Reacher', 
    'HopperBulletEnv': r'Hopper', 
    'HalfCheetahBulletEnv': r'HalfCheetah',
    'Walker2DBulletEnv': r'Walker2D',
    'AntBulletEnv': r'Ant',
    'Humanoid': r'Humanoid'}

LABELS = {
    'hgqn-r1-sum': 'HGQN ($r=1$)',
    'hgqn-r2-sum': 'HGQN ($r=2$)',
    'hgqn-r3-sum': 'HGQN ($r=3$)',
    'bdqn': 'BDQN',
    'dqn': 'DQN',
    'rainbow': 'Rainbow$^\dagger$',
    'ddpg': 'DDPG'}

COLORS = {
    'hgqn-r1-sum': '#ff414d',
    'hgqn-r2-sum': '#2ECC71',
    'hgqn-r3-sum': '#000CF9',
    'bdqn': '#00bcd4',
    'dqn': '#888888',
    'rainbow': '#ffa931',
    'ddpg': '#151680'}

LINESTYLES = {
    'hgqn-r1-sum': '-',
    'hgqn-r2-sum': '-',
    'hgqn-r3-sum': '-',
    'bdqn': '-',
    'dqn': '--',
    'rainbow': '--',
    'ddpg': '--'}

DDPG_PERFORMANCE = {
    'ReacherBulletEnv': 12., 
    'HopperBulletEnv': 1375., 
    'HalfCheetahBulletEnv': 2687.5,
    'Walker2DBulletEnv': 546.875,
    'AntBulletEnv': 2310.,
    'Humanoid': 102.}

width = 12.65
num_figs = 3
hunits_fig = 17
vunits_fig = int(hunits_fig / 1.13333333333)
hunits_between_figs = 5
vunits_between_figs = 5
legend_units = 8
unit_width = width / (num_figs*hunits_fig + (num_figs-1)*hunits_between_figs)
height = unit_width * (2*vunits_fig + vunits_between_figs + legend_units)
fig = plt.figure(figsize=(width, height))

grid = gridspec.GridSpec(2*vunits_fig + vunits_between_figs + legend_units, 
    2*hunits_between_figs + 3*hunits_fig)
ax1 = fig.add_subplot(grid[0:vunits_fig,
    0*hunits_between_figs + 0*hunits_fig:0*hunits_between_figs + 1*hunits_fig])
ax2 = fig.add_subplot(grid[0:vunits_fig, 
    1*hunits_between_figs + 1*hunits_fig:1*hunits_between_figs + 2*hunits_fig])
ax3 = fig.add_subplot(grid[0:vunits_fig, 
    2*hunits_between_figs + 2*hunits_fig:2*hunits_between_figs + 3*hunits_fig])
ax4 = fig.add_subplot(grid[vunits_fig + vunits_between_figs:2*vunits_fig + vunits_between_figs, 
    0*hunits_between_figs + 0*hunits_fig:0*hunits_between_figs + 1*hunits_fig])
ax5 = fig.add_subplot(grid[vunits_fig + vunits_between_figs:2*vunits_fig + vunits_between_figs, 
    1*hunits_between_figs + 1*hunits_fig:1*hunits_between_figs + 2*hunits_fig])
ax6 = fig.add_subplot(grid[vunits_fig + vunits_between_figs:2*vunits_fig + vunits_between_figs, 
    2*hunits_between_figs + 2*hunits_fig:2*hunits_between_figs + 3*hunits_fig])
axl = fig.add_subplot(grid[
    2*vunits_fig + vunits_between_figs:2*vunits_fig + vunits_between_figs + legend_units, 
    0*hunits_between_figs + 0*hunits_fig:2*hunits_between_figs + 3*hunits_fig])

axl.axis('off')
ax = [ax1, ax2, ax3, ax4, ax5, ax6]
for axi in ax: axi.set_axisbelow(True)

for game_id, game in enumerate(GAMES):
  print(game)
  ax[game_id].hlines(y=DDPG_PERFORMANCE[game], xmin=0, xmax=MAX_EPOCH, color=COLORS['ddpg'], 
      linestyle=LINESTYLES['ddpg'], linewidth=2., label=LABELS['ddpg'])

  if game == 'ReacherBulletEnv':
    AGENTS = ['hgqn-r1-sum', 'hgqn-r2-sum', 'bdqn', 'dqn', 'rainbow']
  elif game in ['HopperBulletEnv', 'HalfCheetahBulletEnv', 'Walker2DBulletEnv']:
    AGENTS = ['hgqn-r1-sum', 'hgqn-r2-sum', 'hgqn-r3-sum', 'bdqn', 'dqn', 'rainbow']
  elif game in ['AntBulletEnv', 'Humanoid']:
    AGENTS = ['hgqn-r1-sum', 'bdqn']

  for agent in AGENTS:
    agent_game_data = []
    for seed in SEEDS:
      log_path = BASE_PATH + '/{}/{}/{}/logs'.format(game, agent, seed)
      raw_data, _ = plot_utils.load_statistics(log_path, verbose=False)
      if raw_data is None: 
        iteration_number = plot_utils.get_latest_iteration(log_path)
        raw_data, _ = plot_utils.load_statistics(
            log_path, iteration_number=iteration_number-1, verbose=False)
        assert raw_data is not None
      summarized_data = plot_utils.summarize_data(raw_data, ['eval_episode_returns'])
      agent_game_data.append(summarized_data['eval_episode_returns'][:MAX_EPOCH])

    xs, means, stds = plot_utils.smooth_results(agent_game_data, 5, 1)
    ax[game_id].fill_between(
        range(len(means)), means-stds, means+stds, color=COLORS[agent], alpha=0.1, lw=0.)
    ax[game_id].plot(means, label=LABELS[agent], color=COLORS[agent],
        linestyle=LINESTYLES[agent], linewidth=1.3)

  ax[game_id].set_title(TITLES[game], fontsize=14)
  if game_id in [0,3]: 
    ax[game_id].set_ylabel('Average Score', fontsize=12)
  if game_id >= 3:
    ax[game_id].set_xlabel('Environment Step', fontsize=12)
  ax[game_id].set_xlim([0,MAX_EPOCH])
  ax[game_id].set_xticks([0,100,200,300])
  ax[game_id].set_xticklabels(['0','1M','2M','3M'])

handles, labels = ax[2].get_legend_handles_labels()
new_handles = \
    [handles[0], handles[4], handles[1], handles[5], handles[2], handles[-1], handles[3]]
new_labels = \
    [labels[0], labels[4], labels[1], labels[5], labels[2], labels[-1], labels[3]]
ax[4].legend(new_handles, new_labels, bbox_to_anchor=(0.5, -0.3), loc='upper center', 
    ncol=4, frameon=False, markerscale=2., fontsize=14)

plt.savefig('continuous_control_results.pdf')
print('figure saved')
