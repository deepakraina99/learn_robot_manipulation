import gym
import learn_robot_manipulation
import os
import torch.nn as nn


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

train, test = 1, 0

env = gym.make('TargetReaching-v0')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
# model = PPO('MlpPolicy', env, 
#             learning_rate=3e-5, n_steps=512, 
#             batch_size=64, n_epochs=20, gae_lambda=0.9,
#             use_sde = True, sde_sample_freq=4, clip_range=0.4,
#             policy_kwargs=dict(log_std_init=-2.7,ortho_init=False,activation_fn=nn.ReLU,net_arch=[dict(pi=[256, 256], vf=[256, 256])]), 
#             verbose=1)

if train:
  model.learn(total_timesteps=1000000)
  model.save(os.path.dirname(__file__) + "trained_models/ppo/TargetReaching-v0")
if test:
  model.load(os.path.dirname(__file__) + "trained_models/ppo/TargetReaching-v0")

  obs = env.reset()
  done = False
  while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

  env.close()