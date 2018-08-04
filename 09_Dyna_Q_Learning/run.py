"""
Simplest model-based RL, Dyna-Q
EnvModel用来学习外界环境，与model-free的Q-Learning结合后，Q-Learning就可以在不与外界环境接触的情况下进行学习。

Model-free  RL : Q learning, Sarsa, Police Gradient
Model-based RL : 在model-free的基础上，增加模拟环境的能力(P(s_,r|s,a))

model-based即可以在真实环境中进行学习，也可以在模拟环境中进行学习。而model-free只能等待真实环境的反馈。
model-based可以基于模拟环境进行想象来得出下一步的最好结果？

Advantages of Model-Based RL
    Advantages:
        Can efficiently learn model by supervised learning methods Can reason about model uncertainty
    Disadvantages:
        First learn a model, then construct a value function ⇒ two sources of approximation error

Model-Free 
    - RL No model
    - Learn value function (and/or policy) from real experience 
Model-Based RL (using Sample-Based Planning)
    - Learn a model from real experience
    - Plan value function (and/or policy) from simulated experience
Dyna
    - Learn a model from real experience
    - Learn and plan value function (and/or policy) from real and simulated experience
 
Model-based(Tabel lookup model) + Model-free(Q-Learning) = Tabular Dyna-Q

"""

from maze_env import Maze
from RL_brain import QLearningTable, EnvModel

def update():
    for episode in range(100):
        observation = env.reset()
            
        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))
            
            env_model.store_transition(str(observation), action, reward, observation_)
            # learn 10 more times using the env_model
            for i in range(10):
                model_s, model_a  = env_model.sample_s_a()
                model_r, model_s_ = env_model.get_r_s(model_s, model_a)
                RL.learn(model_s, model_a, model_r, str(model_s_))


            observation = observation_

            if done:
                break
    print('Game Over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env_model = EnvModel(actions=list(range(env.n_actions)))


    env.after(100, update)
    env.mainloop()
