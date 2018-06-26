import numpy as np
import tensorflow as tf
import gym 


np.random.seed(2)
tf.set_random_seed(2) # reproducible


OUTPUT_GRAPH = False
MAX_EPISODE = 1000
DISPLAY_REWARD_THRESHOLD = -200
MAX_EP_STEPS = 200
RENDER= False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01


# Gym environment
env = gym.make('Pendulum-v0')
env.seed(1)
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high


class Actor:
    def __init__(self, sess, n_features, action_bound, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "states")
        self.a = tf.placeholder(tf.float32, None, "actions")
        self.td_error = tf.placeholder(tf.float32, None, "td_error") 

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s,
                    units=20, 
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0.,1.),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='l1')

            mu = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(0., 1.),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='mu')

            sigma = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=tf.nn.softplus,
                    kernel_initializer=tf.random_normal_initializer(0., 1.),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='sigma')
            
            global_step = tf.Variable(0, trainable=False)
            self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
            self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

            self.action = tf.clip_by_value(self.normal_dist.sample(1),
                                            action_bound[0], action_bound[1])

            

        with tf.variable_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            # advantange (TD_errir) guided loss
            self.exp_v = log_prob * self.td_error
            # add cross entropy cost the encourage exploration
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v,global_step)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s:s, self.a:a, self.td_error:td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.sess.run(self.action, {self.s : s})

        return action

class Critic:
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
        self.v_ = tf.placeholder(tf.float32, [1,1], 'v_next')
        self.r = tf.placeholder(tf.float32, None, 'r')


        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0.,1.),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='l1')
            self.v = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0.,1.),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='V')

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA*self.v_ - self.v
            self.loss = tf.square(self.td_error)
            
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})

        td_error,_ = self.sess.run([self.td_error, self.train_op],
                                {self.s:s, self.v_:v_, self.r:r})

        return td_error

sess = tf.Session()
actor = Actor(sess, n_features=N_S, action_bound=[-A_BOUND,A_BOUND],lr=LR_A)
critic = Critic(sess, n_features=N_S, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWrite("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)
        r /= 10



        td_error = critic.learn(s, r, s_)

        actor.learn(s, a, td_error)

        s = s_
        t += 1
        ep_rs.append(r)
        if t >= MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward*0.95 + ep_rs_sum * 0.05

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER=True

            print('episode:', i_episode, " reward:", int(running_reward))
            break



