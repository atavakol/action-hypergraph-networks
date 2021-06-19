from collections import OrderedDict

from dm_control import suite
from dm_env.specs import BoundedArray
import gym
from gym.spaces import Dict
from gym.spaces import Box
import pyglet
import numpy as np


RENDER_KWARGS = {'height': 480, 'width': 640, 'camera_id': 0, 'overlays': (), 'scene_option': None}
RENDER_MODES = {}
RENDER_MODES['human'] = {'show': True, 'return_pixel': False, 'render_kwargs': RENDER_KWARGS}
RENDER_MODES['rgb_array'] = {'show': False, 'return_pixel': True, 'render_kwargs': RENDER_KWARGS}
RENDER_MODES['human_rgb_array'] = {'show': True, 'return_pixel': True, 'render_kwargs': RENDER_KWARGS}


class DMCViewer:
    def __init__(self, width, height):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height
        self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        img = pyglet.image.ImageData(
            self.width, self.height, 'RGB', pixel.tobytes(), pitch=self.pitch)
        img.blit(0, 0)
        self.window.flip()
        
    def close(self):
        self.window.close()


class EnvSpec:
  def __init__(self, id):
    self.id = id


class DMCEnv(gym.Env):
  def __init__(self, domain_name, task_name, task_kwargs=None, seed=None, visualize_reward=False):
    if seed is not None:
        print('Seeding environment with seed', seed)
        if task_kwargs is None:
            task_kwargs = {}
        task_kwargs['random'] = seed
    self.metadata['render.modes'] = list(RENDER_MODES.keys())
    self.viewer = {key: None for key in RENDER_MODES.keys()}
    self.dmc_env = suite.load(domain_name=domain_name, task_name=task_name,
        task_kwargs=task_kwargs, visualize_reward=visualize_reward)
    observation_spec = self.dmc_env.observation_spec()
    assert type(observation_spec) == OrderedDict, observation_spec
    self.observation_space = Dict(
        {k: Box(-np.inf, np.inf, shape=v.shape, dtype='float32') for
            k, v in observation_spec.items()})
    action_spec = self.dmc_env.action_spec()
    assert type(action_spec) == BoundedArray, action_spec
    self.action_space = Box(np.full(action_spec.shape, action_spec.minimum),
                            np.full(action_spec.shape, action_spec.maximum))
    self.spec = EnvSpec(domain_name+'_'+task_name)
    self.aux_env = suite.load(domain_name='hopper', task_name='stand', visualize_reward=True)

  def reset(self):
    self.timestep = self.dmc_env.reset()
    ob = self.timestep.observation
    return ob

  def step(self, action):
    self.timestep = self.dmc_env.step(action)
    ob = self.timestep.observation
    done = False
    info = {'step_type': self.timestep.step_type}
    return ob, self.timestep.reward, done, info
 
  def render(self, mode='human', close=False):
    self.aux_env.physics.render(width=10, height=10, camera_id=0)
    self.pixels = self.dmc_env.physics.render(**RENDER_MODES[mode]['render_kwargs'])
    if close:
        if self.viewer[mode] is not None:
            self._get_viewer(mode).close()
            self.viewer[mode] = None
        return
    elif RENDER_MODES[mode]['show']:
        self._get_viewer(mode).update(self.pixels)
    if RENDER_MODES[mode]['return_pixel']:
        return self.pixels

  def _get_viewer(self, mode):
    if self.viewer[mode] is None:
        self.viewer[mode] = DMCViewer(self.pixels.shape[1], self.pixels.shape[0])
    return self.viewer[mode]


class FlattenedObsEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenedObsEnv, self).__init__(env)
        assert type(env.observation_space) == Dict
        size = sum([int(np.prod(s.shape)) for s in env.observation_space.spaces.values()])
        self.observation_space = Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        return np.concatenate([np.ravel(o) for o in observation.values()])

def make(env_name, seed=None):
    for domain in suite.TASKS_BY_DOMAIN:
        if domain in env_name:
            if 'humanoid_CMU' in env_name:
                domain = 'humanoid_CMU'
            break
    
    if domain != 'lqr':
        task = env_name.split(domain+'_')[-1]
    else:
        task = env_name.split(domain+'_', 1)[-1]
    
    env = DMCEnv(domain, task, seed=seed)
    env = FlattenedObsEnv(env)
    return env
