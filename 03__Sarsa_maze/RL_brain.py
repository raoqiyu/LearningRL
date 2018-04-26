import numpy as np
import pandas as pd

class RL(object):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.9):
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exist(observation)
        # explore or exploit
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # 有些actions有相同的值，避免每次都选同一个action，先打乱
            state_action = state_action.reindex(np.random.permutation(
                                                    state_action.index))
            # exploit
            action = state_action.idxmax()
        else:
            # explore
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                    pd.Series(
                         [0]*len(self.actions),
                        index=self.q_table.columns,
                        name=state,
                        )
                   )

    def learn(self,*args):
        pass

# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.9):
        super(QLearningTable,self).__init__(actions, learning_rate,
                                            reward_decay, e_greedy)

    def learn(self, s, a ,r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

# on-policy
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable,self).__init__(actions, learning_rate,
                                            reward_decay, e_greedy)

    def learn(self, s, a ,r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,
            e_greedy-0.9, trace_decay=0.9):
        super(SarseLambdaTable, self).__init__(actions, learning_rate, reward,
                reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self,)
