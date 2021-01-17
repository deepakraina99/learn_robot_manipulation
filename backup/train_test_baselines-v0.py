import gym
import learn_robot_manipulation
import os
import torch.nn as nn


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

train, test = 1, 0
# train, test = 0, 1

env = gym.make('TargetReaching-v0')
env = DummyVecEnv([lambda: env])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
# model = PPO('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', env, 
            learning_rate=3e-5, n_steps=512, 
            batch_size=64, n_epochs=20, gae_lambda=0.9,
            use_sde = True, sde_sample_freq=4, clip_range=0.4,
            policy_kwargs=dict(log_std_init=-2.7,ortho_init=False,activation_fn=nn.ReLU,net_arch=[dict(pi=[256, 256], vf=[256, 256])]), 
            verbose=1)

if train:
  model.learn(total_timesteps=1e6)
  model.save(os.path.dirname(__file__) + "trained_models/ppo/TargetReaching-v2")
  
if test:
  model.load(os.path.dirname(__file__) + "trained_models/ppo/TargetReaching-v2")

  # obs = env.reset()
  for _ in range(5):
    done = False
    env.render()
    obs = env.reset()
    while not done:
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      # print('Reward:', reward)
      print('info:', info)
      env.render()
      if done:
        obs = env.reset()

    # env.close()