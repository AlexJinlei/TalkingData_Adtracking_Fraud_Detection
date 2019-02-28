import itertools
import math
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
import multiprocessing
from functools import partial
import numba as nb
import gc

def check_time_distance(t_array:np.ndarray, dt_backward:np.timedelta64, dt_forward:np.timedelta64):
    # t_array must be already sorted. (realy?)
    dim = t_array.shape[0]
    count_sum = np.zeros(dim, dtype=np.int32)
    for i in range(dim):
        dts = t_array - t_array[i]
        count_sum[i] = ((dts >= dt_backward) & (dts <= dt_forward)).sum()
    return count_sum

# Only use single core, no @git. Input datatype is np.datetime64 and np.timedelta64.
def check_time_distance_single_core(t_array:np.ndarray, iter_array:np.ndarray=None,  dt_backward:np.timedelta64=np.timedelta64(-3600, 's'), dt_forward:np.timedelta64=np.timedelta64(3600, 's')):
    # iter_array must be a subarray of t_array.
    # t_array must be already sorted.
    count_sum_list = []
    if iter_array is None:
        n_iter = t_array.shape[0]
        iter_array = t_array
    else:
        n_iter = iter_array.shape[0]
    for i in range(n_iter):
        dts = t_array - iter_array[i]
        count_sum_list.append(((dts >= dt_backward) & (dts <= dt_forward)).sum())
    return count_sum_list

# Only use parallel, no @git. Input datatype is np.datetime64 and np.timedelta64.
def check_time_distance_parallel(t_array:np.ndarray, iter_array:np.ndarray=None, num_cpus=2, split_threshold=10000, dt_backward:np.timedelta64=np.timedelta64(-3600, 's'), dt_forward:np.timedelta64=np.timedelta64(3600, 's')):
    if (t_array.shape[0] < split_threshold) | (t_array.shape[0] < num_cpus):
        count_sum_list = check_time_distance_single_core(t_array, iter_array=iter_array, dt_backward=dt_backward, dt_forward=dt_forward)
    else:
        # Split input into shares that equals to num_cpus.
        input_list = np.array_split(t_array, num_cpus) # As iter_array
        cell_func = partial(check_time_distance_single_core, t_array, dt_backward=dt_backward, dt_forward=dt_forward)
        with multiprocessing.Pool(num_cpus) as pool:
            result_list = pool.map(cell_func, input_list)
            # Flatten result_list.
            count_sum_list = [item for sublist in result_list for item in sublist]
    return count_sum_list

# Use single core and @git. Input datatype is int64.
@nb.jit(nopython=True, parallel=True)
def check_time_distance_single_core_jit(t_array:np.ndarray, iter_array:np.ndarray=None,  dt_backward:np.int64=-3600, dt_forward:np.int64=3600):
    # iter_array must be a subarray of t_array.
    # t_array must be already sorted.
    count_sum_list = []
    if iter_array is None:
        n_iter = t_array.shape[0]
        iter_array = t_array
    else:
        n_iter = iter_array.shape[0]
    for i in range(n_iter):
        dts = t_array - iter_array[i]
        count_sum_list.append(((dts >= dt_backward) & (dts <= dt_forward)).sum())
    return count_sum_list

# Use parallel and @git (in a callee function). Input datatype is int64.
def check_time_distance_parallel_jit(t_array:np.ndarray, iter_array:np.ndarray=None, num_cpus=2, split_threshold=10000, dt_backward:np.int64=-3600, dt_forward:np.int64=3600):
    if (t_array.shape[0] < split_threshold) | (t_array.shape[0] < num_cpus):
        count_sum_list = check_time_distance_single_core_jit(t_array, iter_array=iter_array, dt_backward=dt_backward, dt_forward=dt_forward)
    else:
        # Split input into shares that equals to num_cpus.
        input_list = np.array_split(t_array, num_cpus)
        cell_func = partial(check_time_distance_single_core_jit, t_array,  dt_backward=dt_backward, dt_forward=dt_forward)
        with multiprocessing.Pool(num_cpus) as pool:
            result_list = pool.map(cell_func, input_list)
            # Flatten result_list.
            count_sum_list = [item for sublist in result_list for item in sublist]
    return count_sum_list

df_train = pd.read_feather('./data/input/train.feather.small')
df = df_train['click_time'].values

dt_backward_second_int = -5
dt_forward_second_int = 10

dt_backward = np.timedelta64(dt_backward_second_int, 's')
dt_forward = np.timedelta64(dt_forward_second_int, 's')

t_array = df.astype('datetime64[s]')
t_array_int = t_array.view('int64')

funcpool = {'s1': 'check_time_distance_single_core(t_array[:n], dt_backward=dt_backward, dt_forward=dt_forward)', \

            's2': 'check_time_distance_parallel(t_array[:n], split_threshold=100, dt_backward=dt_backward, dt_forward=dt_forward)',\
            
            's3': 'check_time_distance_parallel(t_array[:n], split_threshold=10000000, dt_backward=dt_backward, dt_forward=dt_forward)',\
            
            's4': 'check_time_distance_single_core_jit(t_array_int[:n],  dt_backward=dt_backward_second_int, dt_forward=dt_forward_second_int)',\
            
            's5': 'check_time_distance_parallel_jit(t_array_int[:n], num_cpus=8, split_threshold=100, dt_backward=dt_backward_second_int, dt_forward=dt_forward_second_int)', \
            
            's6': 'check_time_distance_parallel_jit(t_array_int[:n], num_cpus=8, split_threshold=1000000, dt_backward=dt_backward_second_int, dt_forward=dt_forward_second_int)', \
            
            's7': 'check_time_distance(t_array[:n], dt_backward=dt_backward, dt_forward=dt_forward)'}

s = {}
for key in funcpool:
    print(key)
    t = []
    for n in [100, 1000, 10000, 100000]:
        print('n = {}'.format(n))
        t0 = datetime.now()
        s[key] = eval(funcpool[key])        
        t_pass = (datetime.now()-t0).total_seconds()
        t.append(t_pass)
    print(key, t)

'''
s1 [0.001786, 0.026019, 0.81404,  58.272806] check_time_distance_single_core()

s2 [0.111823, 0.120898, 0.521044, 31.931088] check_time_distance_parallel(100)

s3 [0.003071, 0.023971, 0.788398, 58.043189] check_time_distance_parallel(10000000)

s4 [0.248522, 0.0014,   0.122662, 14.207806] check_time_distance_single_core_jit()

s5 [0.757996, 0.846233, 0.761491, 12.504007] check_time_distance_parallel_jit(100)

s6 [0.317704, 0.001661, 0.139156, 14.647803] check_time_distance_parallel_jit(100000)

s7 [0.001878, 0.02237,  0.794313, 56.731372] check_time_distance()
'''

'''
if use @nb.jit(nopython=True, parallel=True)
s1 [0.002147, 0.024318, 0.782169, 59.557441]
s2 [0.112615, 0.113997, 0.534641, 32.489599]
s3 [0.002864, 0.025801, 0.922471, 58.426133]
s4 [0.631846, 0.056971, 0.617884, 9.055291]
s5 [1.887652, 1.671635, 2.071334, 6.431292]
s6 [0.713713, 0.058028, 0.698109, 8.830647]
s7 [0.002323, 0.030639, 0.8476,   57.982084]
'''

    
    
    
    
