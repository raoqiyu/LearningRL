from maze_env import Maze
from RL_brain import SarsaTable, QLearningTable, SarsaLambdaTable

def update_sarsa():
    for episode in range(100):
        observation = env.reset()

        action = RL.choose_action(str(observation))

        while True:
            env.render()

            observation_, reward, done = env.step(action)

            action_ = RL.choose_action(str(observation_))

            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                break
    print('Game Over')
    env.destroy()

def update_Q():
    for episode in range(100):
        observation = env.reset()


        while True:
            env.render()

            action = RL_Q.choose_action(str(observation))

            observation_, reward, done = env.step(action)


            RL_Q.learn(str(observation), action, reward, str(observation_))

            observation = observation_

            if done:
                break

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update_sarsa)
    env.mainloop()
