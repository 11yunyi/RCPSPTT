import numpy as np
import pickle
import matplotlib.pyplot as plt

import environment
import parameters
# import pg_network
import other_agents
import RL_brain
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def categorical_sample(prob_n):
    """
    Sample from categorical distribution,
    specified by a vector of class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    if test_type == 'PG':  # load trained parameters
        tf.reset_default_graph()
        # pg_learner = pg_network.PGLearner(pa)
        rl = RL_brain.PolicyGradient(n_actions=pa.network_output_dim,
                                     network_input_width=pa.network_input_width,
                                     network_input_height=pa.network_input_height,
                                     n_features=pa.network_input_width * pa.network_input_height,
                                     learning_rate=0.02)#初始化一个PG的agent
        rl.load_data(pg_resume)

        # net_handle = open(pg_resume, 'rb')
        # net_params = pickle.load(net_handle)
        # pg_learner.set_net_params(net_params)

    env.reset()
    rews = []

    ob = env.observe()#获得现在环境的观察值

    for _ in range(episode_max_length):#在当前观察值下选择动作

        if test_type == 'PG':
            a = rl.choose_action(ob)

        elif test_type == 'Tetris':
            a = other_agents.get_packer_action(env.machine, env.job_slot)

        elif test_type == 'SJF':
            a = other_agents.get_sjf_action(env.machine, env.job_slot)

        elif test_type == 'Random':
            a = other_agents.get_random_action(env.job_slot)
        
        elif test_type == 'packer':
            a = other_agents.get_packer_sjf_action(env.machine, env.job_slot,0.8)

        ob, rew, done, info = env.step(a, repeat=True)#执行一个动作，获得执行完这个动作之后的观测值

        rews.append(rew)##把所有的单步奖励添加到rews里

        if done: break#如果单个task结束，跳出循环
        if render: env.render()
        # env.render()

    return np.array(rews), info#返回这个episode的奖励轨迹和执行的job轨迹


def launch(pa, pg_resume=None, render=False, plot=False, repre='image', end='no_new_job'):

    # ---- Parameters ----

    test_types = ['Tetris', 'SJF','packer','Random']#测试的类型分为Tetris，SJF和packer

    if pg_resume is not None:
        test_types = ['PG'] + test_types
        print(test_types)
    env = environment.Env(pa, render, repre=repre, end=end)#初始化一个环境

    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []

    for seq_idx in range(pa.num_ex):
        #一组有pa.num_ex数量的序列，每个序列代表一整个task（一整个task就是，比如要完成50个任务的安排）
        print('\n\n')
        print("=============== " + str(seq_idx) + " ===============")

        for test_type in test_types:

            rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            print("---------- " + test_type + " -----------")

            print("total discount reward : \t %s" % (discount(rews, pa.discount)[0]))

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )

            # ------------------------
            # ---- per job stat ----
            # ------------------------

            enter_time = np.array([info.record[i].enter_time for i in range(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in range(len(info.record))])
            job_len = np.array([info.record[i].len for i in range(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(len(info.record))])
            #print('finish_time',finish_time)
            finished_idx = (finish_time >= 0)
            #print('finish_idx',finished_idx)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx])
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    # -- matplotlib colormap no overlap --
    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(color=['grey', 'purple', 'blue', 'green', 'orange', 'red'])
        #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type)

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)
        #plt.show()
        plt.savefig(pg_resume + "_slowdown_fig" + ".pdf")

    return all_discount_rews, jobs_slow_down


def main():
    pa = parameters.Parameters()

    pa.simu_len = 200  # 5000  # 1000
    pa.num_ex = 10  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 1
    pa.discount = 1

    pa.episode_max_length = 20000  # 2000

    pa.compute_dependent_parameters()

    render = True

    plot = True  # plot slowdown cdf

    pg_resume = None
    #pg_resume = 'data/pg_re_discount_1_rate_0.3_simu_len_200_num_seq_per_batch_20_ex_10_nw_10_1450.pkl'
    #pg_resume = 'data/pg_re_1000_discount_1_5990.pkl'
    #pg_resume='data/tem_10.ckpt'
    pa.unseen = True

    launch(pa, pg_resume, render, plot, repre='image', end='all_done')


if __name__ == '__main__':
    main()
