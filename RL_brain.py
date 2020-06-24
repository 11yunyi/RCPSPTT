import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# reproducible
np.random.seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            network_input_height,
            network_input_width,
            n_features,
            learning_rate=0.01,
            output_graph=False,
    ):

        self.n_actions = n_actions
        self.network_input_height=network_input_height
        self.network_input_width=network_input_width
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = 0.95

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        layer = tf.layers.dense(
            inputs= self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            # name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            # name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability


        # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # self.log = -tf.log(self.all_act_prob)
            # self.one_hot = tf.one_hot(self.tf_acts, self.n_actions)
            # self.reduce_sum = self.log * self.one_hot
            # self.neg_log_prob = tf.reduce_sum(self.reduce_sum, axis=1)
            # self.mean = self.neg_log_prob * self.tf_vt
            # self.loss = tf.reduce_mean(self.mean)  # reward guided loss
        self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss


        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        #print('prob:',prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        #print('action:',action)
        return action

    def store_ob(self, s):
        self.ep_obs.append(s)

    def store_action(self, a):
        self.ep_as.append(a)

    def store_adv(self, r):
        self.ep_rs.append(r)

    def learn(self, all_ob, all_action, all_adv):
        # discount and normalize episode reward
        #discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        #_, loss, log, one_hot, reduce_sum, neg_log_prob, mean = self.sess.run([self.train_op, self.loss,
                                                                              #self.log,
                                                                              #self.one_hot,
                                                                             # self.reduce_sum,
                                                                             # self.neg_log_prob,
                                                                             # self.mean],
                                                                        #feed_dict={
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            #self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            #self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            #self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            self.tf_obs: np.array(all_ob),  # shape=[None, n_obs]
            self.tf_acts: np.array(all_action),  # shape=[None, ]
            self.tf_vt: np.array(all_adv),  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # discounted_ep_rs = np.zeros_like(self.ep_rs)
        discounted_ep_rs = np.fabs(np.array(self.ep_rs))

        # running_add = 0
        # for t in reversed(range(0, len(self.ep_rs))):
        # running_add = running_add * self.gamma + self.ep_rs[t]
        # discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_data(self, pg_resume):
        self.saver.save(self.sess, pg_resume + '.ckpt')
        #self.saver.save(self.sess, pg_resume)


    def load_data(self, pg_resume):

        self.saver.restore(self.sess, pg_resume)


