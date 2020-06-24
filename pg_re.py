import time
import numpy as np
import matplotlib.pyplot as plt
import environment
import job_distribution
import slow_down_cdf
import RL_brain
import tensorflow as tf

def init_accums(pg_learner):  # in rmsprop
    accums = []
    params = pg_learner.get_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums


def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in range(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


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


def get_traj(agent, env, episode_max_length):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):
    #for _ in range(5):

        loss = 0
        #print('ob_len:',len(ob[0]))
        a = agent.choose_action(ob)

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob_, rew, done, info = env.step(a, repeat=True)

        # agent.store_transition(ob, a, rew)

        rews.append(rew)

        if done:

            # loss = agent.learn()
            break

        ob = ob_

    # loss = agent.learn()

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': info,
            # 'loss': loss
            }

def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, pa.network_input_height*pa.network_input_width),
        dtype=np.float64)

    timesteps = 0
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob

def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in range(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in range(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in range(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])
    ax.set_prop_cycle(color=['red', 'green', 'blue','yellow'])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_prop_cycle(color=['red', 'green', 'blue','yellow'])
    #ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def get_traj_worker(rl, env, pa):

    trajs = []

    for i in range(pa.num_seq_per_batch):
        traj = get_traj(rl, env, pa.episode_max_length)
        trajs.append(traj)

    all_ob = concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_adv = np.concatenate(advs)

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths
    # all_loss = np.array([traj["loss"] for traj in trajs])

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    return all_eprews, all_eplens, all_slowdown, all_ob, all_action, all_adv


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)#生成一序列的任务，其中包括num_ex个task

    for ex in range(pa.num_ex):#对于每个task

        print("-prepare for env-", ex)

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre=repre, end=end)#初始化一个环境
        env.seq_no = ex
        envs.append(env)

    print("-prepare for worker-")

    rl = RL_brain.PolicyGradient(n_actions=pa.network_output_dim,
                                 network_input_height=pa.network_input_height,
                                 network_input_width=pa.network_input_width,
                                 n_features=pa.network_input_height*pa.network_input_width,
                                 learning_rate=0.001)

    # pg_learner = pg_network.PGLearner(pa)

    # if pg_resume is not None:
    # net_handle = open(pg_resume, 'rb')
    # net_params = cPickle.load(net_handle)
    # pg_learner.set_net_params(net_params)

    # pg_learners.append(pg_learner)

    if pg_resume is not None:
        rl.load_data(pg_resume)

    # accums = init_accums(pg_learners[pa.batch_size])

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None,render=True,
                                                            plot=False, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):#进行每一次迭代
    #for iteration in range(1, 2):

        ex_indices = list(range(pa.num_ex))
        np.random.shuffle(ex_indices)#打乱每一次的所有task

        all_eprews = []#所有迭代次数的rewards
        eprews = []#每次迭代的总rewards
        eplens = []#每次迭代完成所有任务的总时长
        all_slowdown = []#所有迭代的总slowdown

        eprewlist = []
        eplenlist =[]
        slowdownlist =[]
        losslist = []

        ex_counter = 0
        for ex in range(pa.num_ex):

            ex_idx = ex_indices[ex]

            eprew, eplen, slowdown, all_ob, all_action, all_adv = get_traj_worker(rl, envs[ex_idx], pa)
            eprewlist.append(eprew)
            eplenlist.append(eplen)
            slowdownlist.append(slowdown)
            
            
            loss = rl.learn(all_ob, all_action, all_adv)
            losslist.append(loss)

            ex_counter += 1

            if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

                print("\n\n")

                ex_counter = 0

                # all_eprews.extend([r["all_eprews"] for r in result])

                # eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                # eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                # all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))

                # assemble gradients
                # grads = grads_all[0]
                # for i in range(1, len(grads_all)):
                # for j in range(len(grads)):
                # grads[j] += grads_all[i][j]

                # propagate network parameters to others
                # params = pg_learners[pa.batch_size].get_params()

                # rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

                # for i in range(pa.batch_size + 1):
                # pg_learners[i].set_net_params(params)


        timer_end = time.time()

        print("-----------------")
        print("Iteration: \t %i" % iteration)
        print("NumTrajs: \t %i" % len(eprewlist))
        print("NumTimesteps: \t %i" % np.sum(eplenlist))
        print("Loss:     \t %s" % np.mean(losslist))
        print("MaxRew: \t %s" % np.average([np.max(rew) for rew in eprewlist]))
        print("MeanRew: \t %s +- %s" % (np.mean(eprewlist), np.std(eprewlist)))
        print("MeanSlowdown: \t %s" % np.mean([np.mean(sd) for sd in slowdownlist]))
        print("MeanLen: \t %s +- %s" % (np.mean(eplenlist), np.std(eplenlist)))
        print("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in eprewlist]))
        mean_rew_lr_curve.append(np.mean(eprewlist))
        slow_down_lr_curve.append(np.mean([np.mean(sd) for sd in slowdownlist]))

        if iteration % pa.output_freq == 0:

            rl.save_data(pa.output_filename + '_' + str(iteration))

            pa.unseen = True
            #model_file=tf.train.latest_checkpoint('ckpt/')
            #print(model_file)
            #slow_down_cdf.launch(pa, model_file,render=False, plot=True, repre=repre, end=end)
            #slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.ckpt',render=False, plot=True, repre=repre, end=end)
            #pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 30  # 1000
    pa.num_ex = 20  #50 # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20 #20
    pa.output_freq = 10 #50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.4

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()