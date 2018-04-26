"""
Deep Q network
"""

import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped
print(env.action_space, env.observation_space, 
        env.observation_space.high, env.observation_space.low, sep='\n')

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001)

total_steps = 0

for i_episode in range(100):
    observation = env.reset()
    ep_r = 0
    
    while True:
        env.render()

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_
        reward = abs(position - (-0.5)) # r in [0,1]

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward

        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                    'ep_r: ', round(ep_r, 2),
                  'epsilon: ', round(RL.epsilon,2))
            break
        observation = observation_
        total_steps += 1

RL.plot_cost()
