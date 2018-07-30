import tensorflow as tf
import multiprocessing
import threading
import matplotlib.pyplot as plt
import numpy as np
import gym 
import os
import shutil

GAME = 'CartPole-v0'
OUTPUT_GRAPH=True
LOG_DIR='./log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'GlobalNet'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(GAME)
N_S, N_A = env.observation_space.shape[0], env.action_space.n

class A3CNet:
    def __init__(self, scope, globalA3C=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'N_S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32,[None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None,], 'A')
                self.v_target = tf.placeholder(tf.float32, [None,1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob)*tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),axis=1, keep_dims=True)
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalA3C.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalA3C.c_params)]

                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalA3C.a_params))
                    self.update_c_op = OPT_A.apply_gradients(zip(self.c_grads, globalA3C.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., 1.)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')

        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        prob_weights = SESS.run(self.a_prob, feed_dict={
            self.s : s[np.newaxis,:]
            })
        action = np.random.choice(range(prob_weights.shape[1]),
                p=prob_weights.ravel())
        return action


class Worker:
    def __init__(self, name, globalA3C):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.A3C = A3CNet(name, globalA3C)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                a = self.A3C.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                
                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = SESS.run(self.A3C.v, {self.A3C.s:s_[np.newaxis,:]})[0,0]

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s = r + GAMMA*v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.A3C.s : buffer_s,
                        self.A3C.a_his: buffer_a,
                        self.A3C.v_target:buffer_v_target,
                    }
                    self.A3C.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []

                    self.A3C.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99*GLOBAL_RUNNING_R[-1] + 0.01*ep_r)
                    print(self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i"%GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_A3C = A3CNet(GLOBAL_NET_SCOPE)

        workers = []
        for i in range(N_WORKERS):
            i_name = 'W_%i'%i
            workers.append(Worker(i_name, GLOBAL_A3C))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target = job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

