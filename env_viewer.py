import gym
import numpy as np
import learn_robot_manipulation

env = gym.make('TargetReaching-v0')

# observation = env.reset()
# print(observation)
# # env.step(0)
# while True:
#     env.step(0)

while True:
    done = False
    # env.render()
    observation = env.reset()
    action = env.action_space.sample()
    print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))
    
    while not done:
        env.render()
        action = env.action_space.sample()
        # print('action:', action)
        # action=[0,0,0,0,0,0]
        # observation, reward, done, info = env.step(env.action_space.sample())
        observation, reward, done, info = env.step(action)
        # print('Observation:', observation)
        print('Reward:', reward)
        # print('info:', info)