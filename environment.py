import numpy as np
import math
import matplotlib.pyplot as plt
# import theano

import parameters


class Env:
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, render=False, repre='image', end='no_new_job'):             

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist#生成一个job

        self.curr_time = 0#现在的时间点是0

        # set up random seed#固定一个随机值
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:#如果新工作的执行时间序列和资源序列为空
            # generate new work#生成新的工作，这些工作是所有的工作
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)#生成10段序列，每段有50个工作


            ###这段表示，？？？？
            self.workload = np.zeros(pa.num_res)
            for i in range(pa.num_res):
                if i == 0:
                    self.workload[i] = \
                        np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                        float(pa.res_slot) / \
                        float(len(self.nw_len_seqs))
                else:
                    self.workload[i] = \
                        np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                        float(pa.res_slot_havester) / \
                        float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
             #######################################         
                      
            ##把两个数组reshape一下，第一维表示第多少次的序列，第二维表示每次多有工作所占用的时间或者资源          
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
            ##########################################
            
            
        else:#####如果时间和资源的序列不为空
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        self.seq_no = 0  # which example sequence表示在哪个序列
        self.seq_idx = 0  # index in that sequence表示在那个序列里的哪一个任务

        # initialize system 初始化系统
        self.machine = Machine(pa)#用来初始化图表表达的机器
        self.job_slot = JobSlot(pa)#用来初始化等待被安排的工作槽位
        self.job_backlog = JobBacklog(pa)#用来初始化被积压的工作的槽位
        self.job_record = JobRecord()#记录工作
        self.extra_info = ExtraInfo(pa)#记录现在时间点和最后一个到达工作槽的工作的时间差
        ################################
        

    def generate_sequence_work(self, simu_len):#生成整个工作的序列，simu_len代表一个工作循环需要解决simu_len个问题

        nw_len_seq = np.zeros(simu_len, dtype=int)#生成一个数组，代表整个工作序列每个工作需要执行的时间
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)#生成一个二维数组，代表整个工作序列每个工作需要的两种资源

        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes，在每个工作的位置上随机生成一个数字，如果小于新工作率，则在当前位置新生成一个任务

                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq#返回占用时间列表和占用资源列表

    def get_new_job_from_seq(self, seq_no, seq_idx):#从序列里得到一个新的任务
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job
#"""---------------------------------非常复杂，多看几遍----------------------------------------"""
    def observe(self):
        if self.repre == 'image':

            backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
            #用来获得backlog的宽度，代表每个时间点下挤压的任务量的最大可表示量

            image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))
            #初始化一个图表表达

            ir_pt = 0
##############################network的第1 2 段，代表每个资源中clauster和job Queue###############
            for i in range(self.pa.num_res):#对于每种资源
                ###########第一段，代表clauster############
                image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
                ir_pt += self.pa.res_slot
                #####################################
                #########第二段，代表job Queue############
                for j in range(self.pa.num_nw):

                    if self.job_slot.slot[j] is not None:  # fill in a block of work
                        image_repr[: self.job_slot.slot[j].len, ir_pt: ir_pt + self.job_slot.slot[j].res_vec[i]] = 1

                    ir_pt += self.pa.max_job_size
                #######################################
###############################network的第三段，表示任务积压的部分######################
            image_repr[: int(self.job_backlog.curr_size / backlog_width),
                       ir_pt: ir_pt + backlog_width] = 1
            if self.job_backlog.curr_size % backlog_width > 0:
                image_repr[int(self.job_backlog.curr_size / backlog_width),
                           ir_pt: ir_pt + self.job_backlog.curr_size % backlog_width] = 1
            ir_pt += backlog_width
##############################################################################
            image_repr[:, ir_pt: ir_pt + 1] = self.extra_info.time_since_last_new_job / \
                                              float(self.extra_info.max_tracking_time_since_last_job)
 ############network input 每一个时间点最后一个数字代表，最新到达任务的积压时间占到了允许的任务积压时间的程度
            ir_pt += 1
            #self.plot_state()
            assert ir_pt == image_repr.shape[1]

            return image_repr.ravel()[np.newaxis, :]
            #return image_repr
#"""-------------------------------------------------------------------------------------------"""
    def plot_state(self):
        plt.figure("screen", figsize=(20, 5))

        skip_row = 0

        for i in range(self.pa.num_res):

            plt.subplot(self.pa.num_res,
                        1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                        i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0
            if i == 0 :
                plt.xlabel("Resource 1(Transporter)")
                plt.ylabel("Time")
            else:
                plt.xlabel("Resource 2(Havester)")
                plt.ylabel("Time")

            plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
            plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest',origin='upper', vmax=1)

            for j in range(self.pa.num_nw):

                job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
                if self.job_slot.slot[j] is not None:  # fill in a block of work
                    job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

                plt.subplot(self.pa.num_res,
                            1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                            1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

                plt.imshow(job_slot, interpolation='nearest', vmax=1)

                if j == self.pa.num_nw - 1:
                    skip_row += 1

        skip_row -= 1
        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
        backlog = np.zeros((self.pa.time_horizon, backlog_width))

        backlog[: self.job_backlog.curr_size // backlog_width, : backlog_width] = 1
        backlog[self.job_backlog.curr_size // backlog_width, : self.job_backlog.curr_size % backlog_width] = 1

        #plt.subplot(self.pa.num_res,
                    #1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    #self.pa.num_nw + 1 + 1)

        #plt.imshow(backlog, interpolation='nearest', vmax=1)

        #plt.subplot(self.pa.num_res,
                    #1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
                    #self.pa.num_res * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

        extra_info = np.ones((self.pa.time_horizon, 1)) * \
                     self.extra_info.time_since_last_new_job / \
                     float(self.extra_info.max_tracking_time_since_last_job)

        #plt.imshow(extra_info, interpolation='nearest', vmax=1)

        #plt.show()     # manual
        plt.pause(0.0001)  # automatic
        #plt.savefig('aaa'+'.pdf')
###############################################################
######################改每一步的reward就看这里#####################
    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty 

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty 

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty 

        return reward
###################################################################
###################################################################


    def step(self, a, repeat=False):###输入一个动作，获得该动作下的环境观测值！

        status = None #初始化现在的状态

        done = False#初始化是否所有的任务被安排
        reward = 0#初始化这个行为获得的奖励
        info = None#初始化其他信息

        if a == self.pa.num_nw:  # explicit void action 显式的空行为
            #如果动作选择到了空行为，表示当前时间点job Queue里已经没有有效的动作可以选了
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action 隐式的空行为
            #if self.seq_idx >= self.pa.simu_len and \
                    #len(self.machine.running_job) > 0 and \
                    #all(s is None for s in self.job_backlog.backlog):
                #ob, reward, done, info = self.step(a + 1, repeat=True)
                #return ob, reward, done, info
            #else:
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action，如果没有合适的位置安排现在这个任务那么timestep+1
                status = 'MoveOn'
            else:
                status = 'Allocate'

        if status == 'MoveOn':#如果状态是timestep+1的话
            self.curr_time += 1#当前时间点+1
            self.machine.time_proceed(self.curr_time)#timestep +1
            self.extra_info.time_proceed()#现在时间和最后到达Job_Queue时间点的时间差+1
############结束的两种情况，第一种只要任务被执行到一个task的最后一个任务即判定为结束，
#############第二种为所有任务都需要被安排，并且结束，包括running_job和job_backlog 里的任务
            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True
############
            if not done:##如果一个task没结束的话

                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)#从序列里得到一个新的任务
###############这一段表示，从seq取出一个新任务后，判定是直接放入Job_Queue里还是放入backlog里
                    if new_job.len > 0:  # a new job comes

                        to_backlog = True
                        for i in range(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:  # put in new visible job slots
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break

                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()##新来任务后，现在的时间点到最新到来的工作的时间点的差归为0
#################
            reward = self.get_reward()#获取本步的奖励

            # add new jobs
            self.seq_idx += 1

        elif status == 'Allocate':####如果状态时分配任务，表示现在这个时间点还可以分配合适的任务
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]##记录一下被分配的任务
            self.job_slot.slot[a] = None##被分配完后，Job_Queue相应位置变空

            # dequeue backlog
            if self.job_backlog.curr_size > 0:#如果现在Job_Queue里有任务
                self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                #job_backlog也更新一下，把第一个位置的任务放到Job_Queue 里
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1

        ob = self.observe()####获得此刻的观测值
        #self.plot_state()
        info = self.job_record#####执行的这个动作的id放到info里

        if done:#如果整个task都执行完了
            self.seq_idx = 0#恢复seq的位置，重新指回第一个位置

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()
        
        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id#job的编号
        self.res_vec = res_vec#job对两种资源的需求
        self.len = job_len#job执行的时间
        self.enter_time = enter_time#job进入Job Queue的时间
        self.start_time = -1  # not being allocatedjob开始被执行的时间
        self.finish_time = -1#job完成的时间


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.res_slot_havester=pa.res_slot_havester
        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * (self.res_slot,self.res_slot_havester)
        #把clauster里所有时间点的两种资源的可使用情况设为1

        self.running_job = []#初始化一个正在被执行的工作的列表

        # colormap for graphical representation 初始化一个颜色列表，用来表达不同工作
        self.colormap = np.arange(1 / float(pa.job_num_cap), 1, 1 / float(pa.job_num_cap))
        np.random.shuffle(self.colormap)#打乱clormap的顺序

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))
        #初始化一个画布，它一共有三维，第一维表示两种不同的资源
        #第二维表示clauster的时间周期，第三维表示每个时间点上最大可使用的资源

    def allocate_job(self, job, curr_time):
        #用来分配任务，allocated==True代表这个任务已经被安排到avbl_slot中一个合适的位置
        #如果allocated==False代表现在已经没有合适的位置安排这个任务了

        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec
            #执行完这个动作后，所有时间点两种资源可用程度
            

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                # update graphical representation

                used_color = np.unique(self.canvas[:])
                # WARNING: there should be enough colors in the color map
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time > job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        avbl_slot = np.where(self.canvas[res, i, :] == 0)[0]
                        self.canvas[res, i, avbl_slot[: job.res_vec[res]]] = new_color

                break

        return allocated

    def time_proceed(self, curr_time):#stimstep往前前进1格

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        #avbl_slot往前前进一格
        self.avbl_slot[-1, :] = self.res_slot
        #最后一格重新赋满值
        for job in self.running_job:
        #如果现在的时间点大于正在执行的job的结束时间点，则将这个任务移除正在执行的任务列表
            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0


class ExtraInfo:
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):#当有一个新工作来到job_slot时，初始化现在的时间点到最新到来的工作的时间点的差
        self.time_since_last_new_job = 0

    def time_proceed(self):
        #如果现在距离最后到达Job_Queue的时间差没有大于最大允许的时间差，则时间差+1
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print("- Backlog test passed -")


def test_compact_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


def test_image_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in range(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print("- Elapsed time: ", end_time - start_time, "sec -")


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
