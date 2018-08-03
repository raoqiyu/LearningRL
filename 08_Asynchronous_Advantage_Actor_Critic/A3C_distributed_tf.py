import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import gym 
import os
import time

GAME = 'CartPole-v0'
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'GlobalNet'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001

env = gym.make(GAME)
N_S, N_A = env.observation_space.shape[0], env.action_space.n

class A3CNet:
    sess = None
    def __init__(self, scope, opt_a=None, opt_c=None, globalA3C=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
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
                
                self.global_step = tf.train.get_or_create_global_step()
            
            with tf.name_scope('sync'):
                with tf. name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalA3C.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalA3C.c_params)]

                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, globalA3C.a_params),global_step=self.global_step)
                    self.update_c_op = opt_c.apply_gradients(zip(self.c_grads, globalA3C.c_params))

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
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        prob_weights = self.sess.run(self.a_prob, feed_dict={
            self.s : s[np.newaxis,:]
            })
        action = np.random.choice(range(prob_weights.shape[1]),
                p=prob_weights.ravel())
        return action


def work(job_name, task_index, global_ep, lock, r_queue, global_running_r):
    # set work's ip:port (ps/worker)
    cluster = tf.train.ClusterSpec({
        "ps":['localhost:2220','localhost:2221'],
        "worker":['localhost:2222','localhost:2223','localhost:2224','localhost:2225']
        })
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    if job_name == 'ps':
        print('Start Parameter Server:',task_index)
        server.join()
    else:
        t1 = time.time()
        env = gym.make('CartPole-v0').unwrapped
        print('Start worker:', task_index)
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d"%task_index,cluster=cluster)):
            opt_a = tf.train.RMSPropOptimizer(LR_A, name='opt_a')
            opt_c = tf.train.RMSPropOptimizer(LR_C, name='opt_c')
            global_net = A3CNet(GLOBAL_NET_SCOPE)

        local_net = A3CNet('local_ac%d'%task_index, opt_a, opt_c, global_net)
        # set training steps
        hooks = [tf.train.StopAtStepHook(last_step=100000)]
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=True,
                                                hooks=hooks) as sess:
            print('Start worker Session:', task_index)
            local_net.sess = sess
            total_step = 1
            buffer_s, buffer_a, buffer_r = [], [], []
            while (not sess.should_stop()) and (global_ep.value < MAX_GLOBAL_EP):
                s = env.reset()
                ep_r = 0
                while True:
                    a = local_net.choose_action(s)
                    s_, r, done, info = env.step(a)
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
                            v_s_ = sess.run(local_net.v, {local_net.s:s_[np.newaxis,:]})[0,0]
                        buffer_v_target = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA*v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                        feed_dict = {
                            local_net.s : buffer_s,
                            local_net.a_his: buffer_a,
                            local_net.v_target:buffer_v_target,
                        }
                        local_net.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        local_net.pull_global()

                    s = s_
                    total_step += 1
                    if done:
                        if r_queue.empty():
                            global_running_r.value = ep_r
                        else:
                            global_running_r.value = 0.99*global_running_r.value + 0.01*ep_r
                        r_queue.put(global_running_r.value)
                        print(
                            "Task:", task_index,
                            "| Ep:", global_ep.value,
                            "| Ep_r:", global_running_r.value,
                            "| Global_step:", sess.run(local_net.global_step))
                        with lock:
                            global_ep.value += 1
                        break
        print('Worker Done: ', task_index, time.time()-t1)


if __name__ == "__main__":
    # using multiprocessing to create a local cluster with 2 parameter servers and 4 workers
    global_ep = mp.Value('i',0)
    lock = mp.Lock()
    r_queue = mp.Queue()
    global_running_r = mp.Value('d',0)

    jobs = [('ps',0), ('ps',1), ('worker',0), ('worker',1), ('worker', 2), ('worker',3)]

    works = [mp.Process(target=work, args=(job_name, task_index, global_ep, lock, r_queue, global_running_r)) for job_name, task_index in jobs]

    [work.start() for work in works]
    [worker.join() for worker in works[2:]]


    ep_r = []
    while not r_queue.empty():
        ep_r.append(r_queue.get())


    plt.plot(np.arange(len(ep_r)), ep_r)
    plt.title('Distributed training')
    plt.xlabel('Step')
    plt.ylabel('Total moving reward')
    plt.show()
    plt.savefig("A3C_distributed_tf.png")

