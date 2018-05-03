import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
    """
    This SumTree code is modified version and the original code if from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py 

    Story the data with it's priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # parent nodes: capacity; leaves nodes : capacity
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        # record data on the leaf node
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # propagate the change through the tree
        while tree_idx != 0:
            # find the parent node
            tree_idx = (tree_idx-1)//2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index
            0           -> storing priority
           / \
          1    2
        /  \  / \
        3   4 5  6      -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """

        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1     # left child
            cr_idx = cl_idx + 1             # right child
            if cl_idx >= len(self.tree):
                # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:  # search the left sub-tree
                    parent_idx = cl_idx 
                else:                       # search the right sub-tee
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0] # the root

class Memory(object):
    """
    This SumTree code is modified version and the orginal code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py

    store (s, a, r, s_) in SumTree
    """

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition) # set the max p for new p
    
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,),dtype=np.int32),\
                                     np.empty((n, self.tree.data[0].size)),\
                                     np.empty((n,1))
        pri_seg = self.tree.total_p/n # 采样区间
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg *i, pri_seg * (i+1)
            v = np.random.uniform(a,b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta) # 简化后的公式
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon # avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# Deep Q Network off-policy
class DQNPrioritizedReplay:
    def __init__(
            self,n_actions, n_features,
            learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
            replace_target_iter=300, memory_size=500, batch_size=32,
            e_greedy_increment=None, output_graph=False,
            double_q=False, prioritized=True, sess=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        # decide to use double q or not
        self.double_q = double_q
        self.prioritized = prioritized

        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):

        # ---------- build evaluate net ---------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')

        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],
                                        name='Q_target')
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='ISWeights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', 
                    tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                    tf.random_normal_initializer(0., 0.3),  \
                    tf.constant_initializer(0.1)
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1],
                        initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                        collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions],
                        initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],
                        initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

            with tf.variable_scope('loss'):
                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(
                                                self.q_target - self.q_eval),axis=1)
                    self.loss = tf.reduce_mean(self.ISWeights*tf.squared_difference(
                                                self.q_target,self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,
                                                    self.q_eval))

            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ----------- build target net -------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1],
                        initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                        collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions],
                        initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],
                        initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if self.prioritized:
            transition = np.hstack((s, [a,r], s_))
            self.memory.store(transition)
            return

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0

        # forward feed
        actions_value = self.sess.run(self.q_eval, 
                                     feed_dict={self.s:observation})
        action = np.argmax(actions_value)
        self.running_q = self.running_q*0.09 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)
        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget params replaced')

        # sample batch memory from all memory
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                feed_dict={
                    self.s_ : batch_memory[:,-self.n_features:], # fixed parms
                    self.s : batch_memory[:, -self.n_features:], # newest parms
                    })
        # change q_target w.r.t q_eval's action
        q_eval = self.sess.run(self.q_eval, {
            self.s : batch_memory[:, :self.n_features]
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

            q_target[batch_index, eval_act_index] = reward + \
                                                self.gamma * selected_q_next
        """
        batch memory: [s, a, r, s_]
        q_next通过target network输入s_来预测下一个action
        q_eval通过eval network输入s来预测下一个action
        首先通过q_target=q_eval.copy(),将eval的结果作为target
        然后通过target network的预测调整target值
        然后（q_target - q_eval）作为loss，对q_next中相应的相应的action进行BP
        """

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, 
                        self.abs_errors,self.loss],
                        feed_dict={
                            self.s:batch_memory[:,:self.n_features],
                            self.q_target: q_target,
                            self.ISWeights:ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                    feed_dict={
                        self.s: batch_memory[:, :self.n_features],
                        self.q_target: q_target
                     })
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment \
                if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



