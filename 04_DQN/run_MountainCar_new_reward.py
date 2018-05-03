"""
Deep Q network
"""

import gym
import tensorflow as tf
from RL_brain import DoubleDQN
from DQNPrioritizedReplay import DQNPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000
ACTION_SPACE=3
FEATURE_SIZE=2
print(env.action_space, env.observation_space, 
        env.observation_space.high, env.observation_space.low, sep='\n')


sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=FEATURE_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Prioritized_DQN'):
    prioritized_DQN = DQNPrioritizedReplay(
        n_actions=ACTION_SPACE, n_features=FEATURE_SIZE, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, prioritized=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        observation = env.reset()
        while True:
            #env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done:
                reward = 10
            
            RL.store_transition(observation, action, reward, observation_)


            if total_steps > MEMORY_SIZE:   # learning
                RL.learn()

            if done:
                print('episode', i_episode, 'finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_prio = train(prioritized_DQN)

his_natural = train(natural_DQN)

plt.plot(his_natural[0,:], his_natural[1,:]-his_natural[1,0], c='r', label='natural DQN')
plt.plot(his_prio[0,:],his_prio[1,:]-his_prio[1,0], c='b', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()
