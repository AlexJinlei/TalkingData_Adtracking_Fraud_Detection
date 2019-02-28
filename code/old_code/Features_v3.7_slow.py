# Features_v3.7.py
# Restart pool repeatly and unpack multiprocessing resluts (per 1e6).
# Do not use shared memory.
# Use multiprocessing.set_start_method('forkserver') in main code.
# parallel groupby.

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
import os

from collections import defaultdict
from functools import partial

def init_check_time_distance():
    print('Starting {}...'.format(multiprocessing.current_process().name))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def groupby_parallel_atom_return_indices(df, groupby_features=None, time_column_name=None):
    groups_dict = df[groupby_features+[time_column_name]].groupby(by=groupby_features)[time_column_name].groups
    return groups_dict

print('\n{} - start.'.format(datetime.now().strftime(fmt)))
df001_gb = df001.groupby(['os', 'channel'])['click_time'].agg(list)
print('\n{} - end.'.format(datetime.now().strftime(fmt)))
    
def groupby_parallel_atom(df, groupby_features=None, time_column_name=None):
    groups = df[groupby_features+[time_column_name]].groupby(by=groupby_features)[time_column_name]
    df_groups = {name: group.to_frame() for name, group in groups}
    return df_groups 

# @nb.jit will reduce performance.
def check_time_distance(t_array:np.ndarray, dt_backward:np.int32, dt_forward:np.int32):
        # t_array must be already sorted.
        dim = t_array.shape[0]
        count_sum_list = []
        for i in range(dim):
            time_range_start = t_array[i] + dt_backward # dt_backward is negative or zero.
            time_range_end = t_array[i] + dt_forward # dt_forward is positive or zero.
            n = t_array[(t_array >= time_range_start) & (t_array <= time_range_end)].shape[0]
            count_sum_list.append(n)
        return count_sum_list

class FeatureCombination(): # Catogorical features combination.
    def __init__(self, iterable):
        self.features = list(iterable)
        
    def iter_power_set(self, min_n=1, max_n=None):
        if max_n is None:
            max_n = len(self.features)
        assert max_n <= len(self.features), 'max_n must be no greater than the length of the iterable!'
        assert max_n >= min_n, 'max_n must be no smaller than min_n!'
        all_combinations = itertools.chain.from_iterable(itertools.combinations(self.features, n) for n in range(min_n, max_n+1))
        return all_combinations
        
    def power_set(self, min_n=1, max_n=None, include_list=[]):
        all_combinations = list(self.iter_power_set(min_n=min_n, max_n=max_n))
        if include_list:
            all_combinations_include = []
            for combine in all_combinations:
                if set(include_list).issubset(set(combine)):
                    all_combinations_include.append(combine)
        else:
            all_combinations_include = all_combinations
                    
        return tuple(all_combinations_include)

class FeatureGenerator():
    @staticmethod # Old and new have the same performance.
    def __check_time_distance_old(t_array:np.ndarray, dt_backward:np.timedelta64, dt_forward:np.timedelta64, dtype='int32'):
        # t_array must be already sorted.
        dim = t_array.shape[0]
        count_sum_list = []
        for i in range(dim):
            dts = t_array - t_array[i]
            count_sum_list.append(((dts >= dt_backward) & (dts <= dt_forward)).sum().astype(dtype))
        return count_sum_list
        
    @staticmethod
    #@nb.jit(nopython=True, parallel=False) # performance reduced if set parallel=True
    def __check_time_distance(t_array:np.ndarray, dt_backward:np.int32, dt_forward:np.int32):
        # t_array must be already sorted.
        dim = t_array.shape[0]
        count_sum_list = []
        for i in range(dim):
            time_range_start = t_array[i] + dt_backward # dt_backward is negative or zero.
            time_range_end = t_array[i] + dt_forward # dt_forward is positive or zero.
            n = t_array[(t_array >= time_range_start) & (t_array <= time_range_end)].shape[0]
            count_sum_list.append(n)
        return count_sum_list
        
    # Use single core and @git. Input datatype is int64(32). Old and new have the same performance.
    @staticmethod
    @nb.jit(nopython=True) # performance reduced if set parallel=True
    def __check_time_distance_single_core_jit_old(t_array:np.ndarray, iter_array:np.ndarray=None,  dt_backward:np.int64=-3600, dt_forward:np.int64=3600):
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
    
    # Use single core and @git. Input datatype is int64(32).
    @staticmethod
    @nb.jit(nopython=True) # performance reduced if set parallel=True
    def __check_time_distance_single_core_jit(t_array:np.ndarray, iter_array:np.ndarray=None,  dt_backward:np.int64=-3600, dt_forward:np.int64=3600):
        # iter_array must be a subarray of t_array.
        # t_array must be already sorted.
        count_sum_list = []
        if iter_array is None:
            n_iter = t_array.shape[0]
            iter_array = t_array
        else:
            n_iter = iter_array.shape[0]
        for i in range(n_iter):
            time_range_start = iter_array[i] + dt_backward # dt_backward is negative or zero.
            time_range_end = iter_array[i] + dt_forward # dt_forward is positive or zero.
            n = t_array[(t_array >= time_range_start) & (t_array <= time_range_end)].shape[0]
            count_sum_list.append(n)
        return count_sum_list
    
    # Use parallel and @git (in a callee function). Input datatype is int64(32).
    @staticmethod
    def __check_time_distance_parallel_jit(t_array:np.ndarray, iter_array:np.ndarray=None, num_cpus=2, split_threshold=10000, dt_backward:np.int64=-3600, dt_forward:np.int64=3600):
        if (t_array.shape[0] < split_threshold) | (t_array.shape[0] < num_cpus):
            count_sum_list = FeatureGenerator.__check_time_distance_single_core_jit(t_array, iter_array=iter_array, dt_backward=dt_backward, dt_forward=dt_forward)
        else:
            # Split input into shares that equals to num_cpus.
            input_list = np.array_split(t_array, num_cpus)
            cell_func = partial(FeatureGenerator.__check_time_distance_single_core_jit, t_array,  dt_backward=dt_backward, dt_forward=dt_forward)
            with multiprocessing.Pool(num_cpus) as pool:
                result_list = pool.map(cell_func, input_list)
                # Flatten result_list.
                count_sum_list = [item for sublist in result_list for item in sublist]
        return count_sum_list

    # Define a function to calculate feature_click_count.
    @staticmethod
    def __calculate_feature_click_count(groupby_features):
        groupby_features = list(groupby_features)
        new_feature_name = 'count_groupby_' +  '_'.join(groupby_features)
        print('\nGenerating feature: {}'.format(new_feature_name))
        feature_click_count_df = df[groupby_features].groupby(by=groupby_features).size().reset_index().rename(columns={0:new_feature_name})
        return new_feature_name, feature_click_count_df

    @staticmethod
    def add_date_time(df:pd.DataFrame, time_column_name:str, output_dir:str):
        print('Generating "year" feature...')
        df_year = df[time_column_name].dt.year.astype('int16')
        df_year.to_frame().to_feather(os.path.join(output_dir, 'year.feather'))
        del df_year
        gc.collect()
        
        print('Generating "month" feature...')
        df_month = df[time_column_name].dt.month.astype('int8')
        df_month.to_frame().to_feather(os.path.join(output_dir, 'month.feather'))
        del df_month
        gc.collect()
        
        print('Generating "weekday" feature...')
        df_weekday = df[time_column_name].dt.weekday.astype('int8') # The day of the week with Monday=0, Sunday=6.
        df_weekday.to_frame().to_feather(os.path.join(output_dir, 'weekday.feather'))
        del df_weekday
        gc.collect()
        
        print('Generating "day" feature...')
        df_day = df[time_column_name].dt.day.astype('int8')
        df_day.to_frame().to_feather(os.path.join(output_dir, 'day.feather'))
        del df_day
        gc.collect()
        
        print('Generating "hour" feature...')
        df_hour = df[time_column_name].dt.hour.astype('int8')
        df_hour.to_frame().to_feather(os.path.join(output_dir, 'hour.feather'))
        #del df_hour
        #gc.collect()
        
        print('Generating "minute" feature...')
        df_minute = df[time_column_name].dt.minute.astype('int8')
        df_minute.to_frame().to_feather(os.path.join(output_dir, 'minute.feather'))
        #del df_minute
        #gc.collect()
        
        print('Generating "second" feature...')
        df_second = df[time_column_name].dt.second.astype('int8')
        df_second.to_frame().to_feather(os.path.join(output_dir, 'second.feather'))
        #del df_second
        #gc.collect()
        
        print('Generating "hour_of_day" feature...')
        df_hour_of_day = (df_hour + df_minute / 60.0 + df_second / 3600.0).astype('float32')
        df_hour_of_day.to_frame().to_feather(os.path.join(output_dir, 'hour_of_day.feather'))
        del df_hour
        del df_minute
        del df_second
        #del df_hour_of_day
        gc.collect()
        
        print('Generating "hour_of_day_sin" feature...')
        df_hour_of_day_sin = (df_hour_of_day.apply(lambda x: math.sin((x - 12) / 24 * 2 * math.pi))).astype('float32')
        df_hour_of_day_sin.to_frame().to_feather(os.path.join(output_dir, 'hour_of_day_sin.feather'))
        del df_hour_of_day_sin
        gc.collect()
        
        print('Generating "hour_of_day_cos" feature...')
        df_hour_of_day_cos = (df_hour_of_day.apply(lambda x: math.cos((x - 12) / 24 * 2 * math.pi))).astype('float32')
        df_hour_of_day_cos.to_frame().to_feather(os.path.join(output_dir, 'hour_of_day_cos.feather'))
        del df_hour_of_day_cos
        del df_hour_of_day
        gc.collect()
        
        feature_list = ['year', 'month', 'weekday', 'day', 'hour', 'minute', 'second', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos']
        return feature_list

    @staticmethod
    def add_click_time_delta_under_given_feature_combination(df:pd.DataFrame, time_column_name, features_iterable, output_dir, direction=None, dtype='float32'):
        feature_list = []
        print('\nGenerating click time delta features...')
        assert direction is not None, "Please specify keyword 'direction'!"
        assert isinstance(direction, str), "Keyword 'direction' must be a string!"
        assert direction in ['forward', 'backward', 'both'], "Keyword 'direction' must be chosen from ['forward', 'backward', 'both']!"
        print('\ntime_delta direction: {}'.format(direction))
        for groupby_features in features_iterable:
            groupby_features = list(groupby_features)
            all_features = groupby_features + [time_column_name]
            print('\ngroupby_features: {}'.format(groupby_features))
            print('all_features: {}'.format(all_features))
            if direction == 'forward':
                new_feature_name = '_'.join(groupby_features) + '_dt_forward'
                feature_list.append(new_feature_name)
                print('Generating feature: {}'.format(new_feature_name))
                df_tmp = (df[all_features].groupby(groupby_features)[time_column_name].shift(-1) - df[time_column_name]).dt.seconds.astype(dtype)
                df_tmp.to_frame().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
                del df_tmp
                gc.collect()
            elif direction == 'backward':
                new_feature_name = '_'.join(groupby_features) + '_dt_backward'
                feature_list.append(new_feature_name)
                print('Generating feature: {}'.format(new_feature_name))
                df_tmp = (df[time_column_name] - df[all_features].groupby(groupby_features)[time_column_name].shift(1)).dt.seconds.astype(dtype)
                df_tmp.to_frame().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
                del df_tmp
                gc.collect()
            elif direction == 'both':
                new_feature_name_forward = '_'.join(groupby_features) + '_dt_forward'
                new_feature_name_backward = '_'.join(groupby_features) + '_dt_backward'
                feature_list.append(new_feature_name_forward)
                feature_list.append(new_feature_name_backward)
                print('Generating feature: {}'.format(new_feature_name_forward))
                print('Generating feature: {}'.format(new_feature_name_backward))
                df_tmp_forward = (df[all_features].groupby(groupby_features)[time_column_name].shift(-1) - df[time_column_name]).dt.seconds.astype(dtype)
                df_tmp_forward.to_frame().to_feather(os.path.join(output_dir, new_feature_name_forward + '.feather'))
                del df_tmp_forward
                gc.collect()
                df_tmp_backward = (df[time_column_name] - df[all_features].groupby(groupby_features)[time_column_name].shift(1)).dt.seconds.astype(dtype)
                df_tmp_backward.to_frame().to_feather(os.path.join(output_dir, new_feature_name_backward + '.feather'))
                del df_tmp_backward
                gc.collect()
        return feature_list

    @staticmethod
    def add_click_count_under_given_feature_combination(df:pd.DataFrame, features_iterable, output_dir, dtype='int32'):
        feature_list = []
        print('\nGenerating the count of each groups grouped by given feature combinations...')
        for groupby_features in features_iterable:
            groupby_features = list(groupby_features)
            new_feature_name = 'count_groupby_' +  '_'.join(groupby_features)
            feature_list.append(new_feature_name)
            print('\nGenerating feature: {}'.format(new_feature_name))
            feature_click_count = df[groupby_features].groupby(by=groupby_features).size().reset_index().rename(columns={0:new_feature_name}).astype(dtype)
            # Merge feature_click_count using database-style join operation.
            df_tmp = df.merge(feature_click_count, on=groupby_features, how='left')
            df_tmp[new_feature_name].to_frame().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
            # Clean memory.
            del feature_click_count # Removes a reference, which decrements the reference count on the value. 
            gc.collect() # Free memory of the unreferenced (reference count = 0) value.
        return df
    
    @staticmethod
    def add_click_count_under_given_feature_combination_within_time_range(df:pd.DataFrame, time_column_name, features_iterable, output_dir, dt_range=[], dtype='int32'): # dt_range = [dt_backward, dt_forward](in seconds)
        assert isinstance(dt_range, list), 'dt_range must be a list!'
        feature_list = []
        # Check if dt_range is specified in argument list.
        if not dt_range: # dt_range is not specified. Calculate the total click count.
            print('\nGenerating the count of each groups grouped by given feature combinations...')
            
            # Check how many cores on cpu.
            num_cpus = multiprocessing.cpu_count()
            print('\nTotol CPU cores on this node = {}'.format(num_cpus))
            num_cpus_use = num_cpus - 1
            print('Use {} cores for multiprocessing.'.format(num_cpus_use))
            # Create a multiprocessing pool with safe_lock.
            pool = multiprocessing.Pool(processes=num_cpus_use)
            # Create a list to save result.
            multiprocessing_results = []
            
            # Run calculate_feature_click_count() on every feature combinations.
            for groupby_features in features_iterable:
                # Return value: (new_feature_name, feature_click_count_df)
                result_temp = pool.apply_async(FeatureGenerator.__calculate_feature_click_count, args=(groupby_features,))
                multiprocessing_results.append(result_temp)
            
            '''
            # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the for loop goes through. The for loop is non-blocked, so it finish in seconds.
            # Wait for all worker processes in pool to finish their work.
            while len(pool._cache)!=0:
                print('\n{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
                time.sleep(0.2)
            print('\n{} - Done. There is 0 worker processes in pool.'.format(time.ctime()))
            '''
            
            # Close pool, prevent new worker process from joining.
            pool.close()
            # Block caller process until workder processes terminate.
            pool.join()
            
            # Unpack multiprecessing_results.
            print('\nMerging new feature to original data frame...')
            for one_result in multiprocessing_results:
                new_feature_name, feature_click_count_df = one_result.get()
                feature_list.append(new_feature_name)
                # Merge feature_click_count using database-style join operation.
                df_tmp = df.merge(feature_click_count_df, on=groupby_features, how='left')
                print('Saving {}'.format(new_feature_name))
                df_tmp[new_feature_name].astype(dtype).to_frame().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
                del feature_click_count_df
                del df_tmp
                gc.collect()
            print('\nDone.')
            
            # Clean memory.
            del multiprocessing_results
            gc.collect()
            
        else: # dt_range is specified.
            fmt = '%Y-%m-%d %H:%M:%S' # Time string format.
            assert len(dt_range)==2, 'dt_range must be a 2 elements list!'
            dt_backward_int_seconds, dt_forward_int_seconds = int(dt_range[0]), int(dt_range[1])
            assert (dt_backward_int_seconds <= 0), 'The first element in dt_range must be <= 0!'
            assert (dt_forward_int_seconds >= 0), 'The second element in dt_range must be >= 0!'
            
            # Check how many cores on cpu.
            num_cpus = multiprocessing.cpu_count()
            print('\nTotol CPU cores on this node = {}'.format(num_cpus))
            num_cpus_use = num_cpus - 1
            print('Use {} cores for multiprocessing.'.format(num_cpus_use))
            
            pool_volume = int(5e5)
            print('\nSet maximum pool volume to {}.'.format(pool_volume))
            max_pool_queue_length = int(5e4) # To many worker process in queue will eat up memory.
            print('\nSet maximum queue length to {}.'.format(max_pool_queue_length))
            wait_time = 3 # Wait seconds when worker process in pool exceed max_pool_queue_length.
            
            for groupby_features in features_iterable:
                # Get groupby feature list and generate new feature name.
                groupby_features = list(groupby_features)
                new_feature_name = 'count_groupby_' + '_'.join(groupby_features) + '_in_time_range_' + str(dt_backward_int_seconds) + '_' + str(dt_forward_int_seconds) + '_seconds'
                feature_list.append(new_feature_name)
                print('\n\n{} - Generating feature: {}.'.format(datetime.now().strftime(fmt), new_feature_name))
            
                # Create a list to save result.
                all_indices_list = []
                all_counts_list = []
                # Create a list to save unpacked multiprocessing results.
                multiprocessing_results = []
                # Dataframe to organize final result.
                counts_in_time_range_df = pd.Series([], name=new_feature_name, dtype=dtype).to_frame()
                
                '''
                # Group by groupby_features, then only keep the click_time column.
                print('\n{} - Grouping...'.format(datetime.now().strftime(fmt))) 
                groups = df[groupby_features+[time_column_name]].groupby(by=groupby_features)[time_column_name]
                n_groups = len(groups)
                print('\n{} - Done. Number of groups: {}.'.format(datetime.now().strftime(fmt), n_groups))
                '''
                
                
                df_list = np.array_split(df, num_cpus_use)
                
                
                partial_groubpy_parallel_atom = partial(groupby_parallel_atom, groupby_features=groupby_features, time_column_name=time_column_name)
                
                pool = multiprocessing.Pool(processes=num_cpus_use)
                d = pool.map(partial_groubpy_parallel_atom, df_list)
                pool.close()
                pool.join()
                del pool
                gc.collect()
                
                """merged_dict = {}
                for a in d: # you can list as many input dicts as you want here
                    for key, value in a.items():
                        if key not in merged_dict:
                            merged_dict[key] = value
                        else:
                            merged_dict[key] = merged_dict[key].append(value)"""
                            
                merged_groups_dict = {}
                for a in d: # you can list as many input dicts as you want here
                    for name, df in a.items():
                        if name not in merged_groups_dict:
                            merged_groups_dict[name] = df
                        else:
                            merged_groups_dict[name] = pd.concat([merged_groups_dict[name], df]).sort_index()
                
                n_groups = len(merged_groups_dict)
                
                
                # Create a multiprocessing pool.
                print('\n{} - Creating multiprocessing pool...'.format(datetime.now().strftime(fmt)))
                pool = multiprocessing.Pool(processes=num_cpus_use)
                print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
                
                """print('\n{} - Creating groups...'.format(datetime.now().strftime(fmt)))
                groups = {name: df_t_array.loc[group_indices] for name, group_indices in merged_dict.items()}
                print('\n{} - Done.'.format(datetime.now().strftime(fmt)))"""
                
                print('\n{} - Creating {} worker processes...'.format(datetime.now().strftime(fmt), n_groups))
                # Loop all group in groups to calculate counts within time range in each group.
                for i, (name, group) in enumerate(merged_groups_dict.items()): # group should be a Series.
                
                #for i, (name, group_indices) in enumerate(merged_dict.items()): # group should be a Series.
                    if (not (i % 10000)) & (i > 0):
                        print('\n{} - {}/{} worker processes added.'.format(datetime.now().strftime(fmt), i, n_groups))
                    
                    if (not (i % pool_volume)) & (i > 0):
                        print('\n{} - i = {}. Reached maximum pool volume ({}), unpacking multiprecessing results...'.format(datetime.now().strftime(fmt), i, pool_volume))
                        # Wait for all worker processed to finish before restarting pool.
                        while len(pool._cache)!=0:
                            print('\n{} - Wait for all worker processes to finish before restarting pool. \nThere are {} worker processes left in pool.'.format(datetime.now().strftime(fmt), len(pool._cache)))
                            time.sleep(1)
                        print('\n{} - Done. There is 0 worker processes left in pool.'.format(datetime.now().strftime(fmt)))
                        
                        # Unpack multiprecessing_results.
                        print('\n{} - Unpacking multiprecessing results...'.format(datetime.now().strftime(fmt)))
                        #print('len(multiprocessing_results) = {}'.format(len(multiprocessing_results)))
                        for one_result in multiprocessing_results:
                            # Append each result in counts list.
                            all_counts_list += list(np.array(one_result.get(), dtype=dtype))
                        print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
                        
                        # Create dataframe from current result and append to counts_in_time_range_df.
                        print('\n{} - Creating temporary dataframe from current indices and counts lists...'.format(datetime.now().strftime(fmt)))
                        df_tmp = pd.Series(all_counts_list, index=all_indices_list, name=new_feature_name, dtype=dtype).to_frame()
                        # Append temp series to dataframe.
                        print('\n{} - Appending results to counts_in_time_range_df...'.format(datetime.now().strftime(fmt)))
                        counts_in_time_range_df = counts_in_time_range_df.append(df_tmp)
                        print('\n{} - Results appended.'.format(datetime.now().strftime(fmt)))
                        
                        # Reset lists, and clear memory.
                        print('\n{} - Reseting lists...'.format(datetime.now().strftime(fmt)))
                        all_indices_list = []
                        all_counts_list = []
                        multiprocessing_results = []
                        del df_tmp
                        print('\n{} - Objects collected = {}.'.format(datetime.now().strftime(fmt), gc.collect()))
                    
                    # Wait all worker precesses to finishe.
                    if len(pool._cache) >= max_pool_queue_length:
                        print('\n{} - Reached maximum pool queue length (max_pool_queue_length = {}), wait {} second...'.format(datetime.now().strftime(fmt), max_pool_queue_length, wait_time))
                        time.sleep(wait_time)
                        print('\n{} - processed = {}, in_pool = {}, total = {}'.format(datetime.now().strftime(fmt), len(multiprocessing_results), len(pool._cache), n_groups))
                    
                    # Parallelly process each group.
                    #print(group_indices)
                    #print(df[time_column_name].loc[group_indices].values)

                    #t_array = df_t_array.loc[group_indices].values
                    
                    multiprocessing_results.append(pool.apply_async(check_time_distance, args=(group.values, dt_backward_int_seconds, dt_forward_int_seconds)))
                    # Keep track of indices of each group.
                    #all_indices_list += group.index.tolist() # Keep records of corresponding index.
                    all_indices_list += group.index.tolist()
                
                print('\n{} - Iterations for {} is finished.'.format(datetime.now().strftime(fmt), new_feature_name))
                # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the for loop goes through. The for loop is non-blocked, so it finish in seconds.
                # Wait for all worker processes in pool to finish their work.
                
                while len(pool._cache)!=0:
                    print('\n{} - Waiting... There are {} worker processes left in pool.'.format(datetime.now().strftime(fmt), len(pool._cache)))
                    time.sleep(1)
                print('\n{} - Done. There is 0 worker processes left in pool.'.format(datetime.now().strftime(fmt)))
                
                # Close pool, prevent new worker process from joining.
                print('\n{} - Closing pool...'.format(datetime.now().strftime(fmt)))
                pool.close()
                # Block caller process until workder processes terminate.
                pool.join()
                del pool
                
                # Unpack multiprecessing_results.
                print('\n{} - Unpacking the rest multiprecessing results...'.format(datetime.now().strftime(fmt)))
                for one_result in multiprocessing_results:
                    # Append each result in counts list.
                    all_counts_list += list(np.array(one_result.get(), dtype=dtype))
                print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
                
                # Create dataframe from current result and append to counts_in_time_range_df.
                print('\n{} - Creating temporary dataframe from current indices and counts lists...'.format(datetime.now().strftime(fmt)))
                df_tmp = pd.Series(all_counts_list, index=all_indices_list, name=new_feature_name, dtype=dtype).to_frame()
                # Append temp series to dataframe.
                print('\n{} - Appending results to counts_in_time_range_df...'.format(datetime.now().strftime(fmt)))
                counts_in_time_range_df = counts_in_time_range_df.append(df_tmp)
                print('\n{} - Results appended.'.format(datetime.now().strftime(fmt)))
                
                '''
                # Merge according to index.
                print('\n{} - Merging new feature to original dataframe...'.format(datetime.now().strftime(fmt)))
                df_tmp = df.merge(counts_in_time_range_df, left_index=True, right_index=True, how='left')
                print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
                print('\n{} - Saving new feature as feather file...'.format(datetime.now().strftime(fmt)))
                df_tmp[new_feature_name].astype(dtype).to_frame().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
                print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
                '''
                
                # Sort index and save to feather file.
                counts_in_time_range_df.sort_index().to_feather(os.path.join(output_dir, new_feature_name + '.feather'))
                
                # Clean memory.
                print('\n{} - Clearning memory...'.format(datetime.now().strftime(fmt))) 
                del all_indices_list
                del all_counts_list
                del multiprocessing_results
                del df_tmp
                del counts_in_time_range_df
                print('\n{} - Objects collected = {}.'.format(datetime.now().strftime(fmt), gc.collect()))
                       
                print('\n{} - Feature {} is created and saved.'.format(datetime.now().strftime(fmt), new_feature_name))

        return feature_list
    
    @staticmethod
    def add_unique_count_of_feature_under_given_feature_combination(df:pd.DataFrame, features_iterable, objective_features:list, output_dir:str, dtype='int32'):
        feature_list = []
        print('\nGenerating the unique count of feature under given feature combinations...')
        print('objective_features: {}'.format(objective_features))
        for groupby_features in features_iterable:
            groupby_features = list(groupby_features)
            print('\ngroupby_features: {}'.format(groupby_features))
            objective_features_without_groupby_features = list(set(objective_features) - set(groupby_features))
            print('objective_features_without_groupby_features: {}'.format(objective_features_without_groupby_features))
            if not objective_features_without_groupby_features: # bool([]) is false.
                print('objective_features_without_groupby_features is empty. Skip.')
                continue
            all_features = list(set(objective_features).union(set(groupby_features)))
            rename_dict = {} # To be used to rename columns.
            for feature in objective_features_without_groupby_features:
                rename_dict[feature] = 'unique_count_' + feature + '_groupby_' + '_'.join(groupby_features)
                feature_list.append(rename_dict[feature])
            
            feature_click_count_unique = df[all_features].groupby(by=groupby_features)[objective_features_without_groupby_features].nunique().reset_index().rename(columns=rename_dict).astype(dtype)
            # Merge click_count using database-style join operation.
            df_tmp = df.merge(feature_click_count_unique, on=groupby_features, how='left')
            df_tmp[rename_dict[feature]].to_frame().to_feather(os.path.join(output_dir, rename_dict[feature] + '.feather'))
            # Clean memory.
            del all_features
            del rename_dict
            del feature_click_count_unique
            del df_tmp
            gc.collect()
        return feature_list


    
    
    
    
    
    

