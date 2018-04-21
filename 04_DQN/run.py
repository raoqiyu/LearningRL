from maze_env import Maze
from RL_brain import DeepQNetwork

def update_Q():
    step = 0
    for episode in range(100):
        observation = env.reset()

        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)
            
            # DeepQ 需要历史信息进行训练
            RL.store_transition(observation, action, reward, observation_)
            observation = observation_

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            if done:
                break
            step += 1
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, 2,
                        learning_rate=0.01, reward_decay=0.9,
                        e_greedy=0.9, replace_target_iter=200,
                        memory_size=2000)

    env.after(100, update_Q)
    env.mainloop()
