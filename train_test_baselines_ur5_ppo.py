import gym
import learn_robot_manipulation
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
save_model_dir = os.path.dirname(__file__) + "trained_models/ppo/" + "TargetReachingUR5"
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

total_timesteps=int(1e6)
tests = 10

train, test = 1, 0
# train, test = 0, 1

if train:
  env = gym.make('TargetReachingUR5-v0')
  env = Monitor(env, log_dir)

  env = DummyVecEnv([lambda: env])
  # Automatically normalize the input features and reward
  env = VecNormalize(env, norm_obs=True, norm_reward=True,
                    clip_obs=10.)
  model = PPO('MlpPolicy', env, verbose=1)
  # model = PPO('MlpPolicy', env, 
  #             learning_rate=3e-5, n_steps=512, 
  #             batch_size=64, n_epochs=20, gae_lambda=0.9,
  #             use_sde = True, sde_sample_freq=4, clip_range=0.4,
  #             policy_kwargs=dict(log_std_init=-2.7,ortho_init=False,activation_fn=nn.ReLU,net_arch=[dict(pi=[256, 256], vf=[256, 256])]), 
#             verbose=1)
  model.learn(total_timesteps)
  model.save(save_model_dir + "/TargetReachingUR5-v0")
  stats_path = save_model_dir + "/vec_normalize.pkl"
  env.save(stats_path)

  plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "PPO TargetReaching UR5")
  plt.savefig(save_model_dir + '/Rewards.png')
  plt.show()
  
if test:
  model = PPO.load(save_model_dir + "/TargetReachingUR5-v0")
  # Load the saved statistics
  env = gym.make('TargetReachingUR5-v0')
  env = DummyVecEnv([lambda: env])
  stats_path = save_model_dir + "/vec_normalize.pkl"
  env = VecNormalize.load(stats_path, env)
  #  do not update them at test time
  env.training = False
  # reward normalization is not needed at test time
  env.norm_reward = False
  success = 0
  # obs = env.reset()
  env.seed(10)
  for _ in range(tests):
    done = False
    env.render()
    obs = env.reset()
    while not done:
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      # print('Reward:', reward)
      env.render()
      if info[0]['task_success']:
        print('Reached Goal !!')
        done = True
        success +=1
        obs = env.reset()
  print('success: {}/{}'.format(success, tests))
    # env.close()