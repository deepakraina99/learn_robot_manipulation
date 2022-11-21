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
from stable_baselines3.common.callbacks import EvalCallback

save_model_dir = os.path.dirname(__file__) + "trained_models/ppo/" + "TargetReachingUR5_taskspace_vs_jointspace_3e6"
if not os.path.exists(save_model_dir):
  os.makedirs(save_model_dir)
log_dir = save_model_dir

total_timesteps=int(2e6)
tests = 10
# 
# train, test = 1, 0
train, test = 0, 1

if train:
  env = gym.make('TargetReachingUR5-v0')
  env = Monitor(env, log_dir)

  env = DummyVecEnv([lambda: env])
  # Automatically normalize the input features and reward
  env = VecNormalize(env, norm_obs=True, norm_reward=True,
                    clip_obs=10.)
  # Use deterministic actions for evaluation
  eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=1000,
                             deterministic=True, render=False)  
  # model = PPO('MlpPolicy', env, verbose=1)
  # model = PPO('MlpPolicy', env,
  #             learning_rate=3e-5,
  #             verbose=1)
  # model = PPO('MlpPolicy', env,
  #             learning_rate=3e-5,
  #             use_sde=True, sde_sample_freq=4,
  #             verbose=1)             
  model = PPO('MlpPolicy', env,
              learning_rate=3e-5,
              use_sde=True, sde_sample_freq=3,
              policy_kwargs=dict(activation_fn=nn.Tanh),
              verbose=0)     
  # model = PPO('MlpPolicy', env, 
  #             learning_rate=3e-5, n_steps=512, 
  #             batch_size=64, n_epochs=20, gae_lambda=0.9,
  #             use_sde = True, sde_sample_freq=4, clip_range=0.4,
  #             policy_kwargs=dict(log_std_init=-2.7,ortho_init=False,activation_fn=nn.ReLU,net_arch=[dict(pi=[256, 256], vf=[256, 256])]), 
  #           verbose=1)
  model.learn(total_timesteps, callback=eval_callback)
  model.save(save_model_dir + "/TargetReaching-v0")
  stats_path = save_model_dir + "/vec_normalize.pkl"
  env.save(stats_path)

  plot_results([log_dir], total_timesteps, results_plotter.X_TIMESTEPS, "PPO TargetReaching UR5")
  plt.savefig(save_model_dir + '/Rewards.png')
  plt.show()
  
  def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


  def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(save_model_dir + '/Learning_curve.png')
    plt.show()

  plot_results(log_dir)
  
if test:
  # model = PPO.load(save_model_dir + "/best_model")
  model = PPO.load(save_model_dir + "/TargetReaching-v0")
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
  # env.seed(100)
  for _ in range(tests):
    done = False
    env.render()
    obs = env.reset()
    while not done:
      action, _states = model.predict(obs)
      print('action:', action)
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