import numpy as np


class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res#环境中任务的种类
        self.max_nw_size = max_nw_size#新任务需求的最大资源（每个任务需求的资源不能超过这个值）
        self.job_len = job_len#新任务最大的执行时间，新任务执行时间不能超过这个值

        self.job_small_chance = 0.8#小任务的概率是0.8

        self.job_len_big_lower = job_len * 2 / 3#一个大任务执行时间的下界是job_len的2/3
        self.job_len_big_upper = job_len#一个大任务执行时间的上界是job_len

        self.job_len_small_lower = 1#一个小任务执行时间的下界是1
        self.job_len_small_upper = job_len / 5#一个小任务执行时间的上界是len_job的1/5

        self.dominant_res_lower = 1*max_nw_size / 3#一个任务占主导地位的需求资源的的下界是max_nx_size的1/2
        self.dominant_res_upper = max_nw_size#一个任务占主导地位的需求资源的的上界是max_nx_size

        self.other_res_lower = 1#一个任务占非主导地位的需求资源的的下界是1
        self.other_res_upper = max_nw_size / 5#一个任务占非主导地位的需求资源的的上界是max_nx_size的1/5

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 1)  # same length in every dimension

        nw_size = np.zeros(self.num_res)

        for i in range(self.num_res):
            nw_size[i] = np.random.randint(1, self.max_nw_size + 1)

        return nw_len, nw_size

    def bi_model_dist(self):#这是用来生成一个job的方法

        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job，生成一个随机数，如果小于0.8，则生成一个小任务
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)#随机生成小任务的执行时间
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)#随机生成一个大任务的执行时间

        

        # -- job resource request --
        nw_size = np.zeros(self.num_res)#生成一个数组，用来记录一个job的两种资源的需求情况
        dominant_res = 0#用来表示这个任务占主导地位的资源是第一种资源（Transporter）
        for i in range(self.num_res):#给这个任务随机生成两种资源的需求情况
            if i == dominant_res:
                nw_size[i] = np.random.randint(self.dominant_res_lower,
                                               self.dominant_res_upper + 1)
            else:
                nw_size[i] = np.random.randint(self.other_res_lower,
                                               self.other_res_upper + 1)
                    
                #nw_size[i] = 1
        return nw_len, nw_size#返回值是一个数组，第一个数是这个任务的执行时间，第二个数是一个数组，代表两种资源的需求


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    nw_dist = pa.dist.bi_model_dist

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)

    for i in range(simu_len):

        if np.random.rand() < pa.new_job_rate:  # a new job comes

            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])

    return nw_len_seq, nw_size_seq
