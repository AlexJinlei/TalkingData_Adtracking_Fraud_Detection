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

'''
O(n^2) very slow. Keep it for double check check_time_distance().
def check_time_distance_old(t_array:np.ndarray, dt_backward:np.timedelta64, dt_forward:np.timedelta64):
    # t_array must be already sorted.
    dim = len(t_array)
    dist_matrix = np.zeros([dim, dim])
    for row in range(dim):
        for column in range(dim):
            dt = t_array[column] - t_array[row]
            dist_matrix[row, column] = (dt >= dt_backward) & (dt <= dt_forward)
    return dist_matrix.sum(axis=1)
'''

'''
# This is even slower than check_time_distance().
def check_time_distance_sorted_t_array(t_array:np.ndarray, dt_backward:np.timedelta64, dt_forward:np.timedelta64):
    # t_array must be already sorted.
    dim = t_array.shape[0]
    count_sum_list = []
    for i in range(dim):
        sub_t_array_start_index, sub_t_array_end_index = i, i
        #print('\ni = {}/{}'.format(i,dim))
        # Check backward.
        #print('Check backward')
        while(t_array[sub_t_array_start_index] - t_array[i] >= dt_backward):
            #print(sub_t_array_start_index, sub_t_array_end_index)
            sub_t_array_start_index -= 1
            if(sub_t_array_start_index < 0):
                sub_t_array_start_index += 1
                break
        # Check forward.
        #print('Check forward')
        while(t_array[sub_t_array_end_index] - t_array[i] <= dt_forward): # max index is dim - 1.
            #print(sub_t_array_start_index, sub_t_array_end_index)
            sub_t_array_end_index += 1
            if(sub_t_array_end_index >= dim):
                sub_t_array_end_index -= 1
                break
        # Count total and append to list.
        count_sum_list.append(sub_t_array_end_index - sub_t_array_start_index + 1)
    return count_sum_list
'''

def check_time_distance(t_array:np.ndarray, dt_backward:np.timedelta64, dt_forward:np.timedelta64):
    # t_array must be already sorted.
    dim = t_array.shape[0]
    count_sum_list = []
    for i in range(dim):
        dts = t_array - t_array[i]
        count_sum_list.append(((dts >= dt_backward) & (dts <= dt_forward)).sum())
    return count_sum_list

# Use single core and @git. Input datatype is int64.
@nb.jit(nopython=True, parallel=False)
def check_time_distance_jit(t_array:np.ndarray, dt_backward, dt_forward):
    count_sum_list = []
    n_iter = t_array.shape[0]
    for i in range(n_iter):
        dts = t_array - t_array[i]
        count_sum_list.append(((dts >= dt_backward) & (dts <= dt_forward)).sum())
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
    
def calculate_feature_click_count_within_time_range(groups, dt_backward, dt_forward, new_feature_name):
    counts_series = pd.Series([], name=new_feature_name)
    for group_name, group in groups: # one_group should be a Series.
        group_name = group.index # All index in current group.
        t_array = one_group.dt.to_pydatetime() # All timestamps in current group.
        counts_in_one_group_list = []
        for t in t_array:
            dt = t_array - t
            is_in_time_range = (dt >= dt_backward) & (dt <= dt_forward)
            counts_in_time_range = is_in_time_range.sum()
            counts_in_one_group_list.append(counts_in_time_range)
        # When the loop within one group is finished, append the result to counts_series.
        counts_series = counts_series.append(pd.Series(counts_in_one_group_list, index=index_one_group, name=new_feature_name))
    return counts_series

# Define a function to calculate feature_click_count.
def calculate_feature_click_count(groupby_features):
    groupby_features = list(groupby_features)
    new_feature_name = 'count_groupby_' +  '_'.join(groupby_features)
    print('\nAdding feature: {}'.format(new_feature_name))
    feature_click_count_df = df[groupby_features].groupby(by=groupby_features).size().reset_index().rename(columns={0:new_feature_name})
    return feature_click_count_df, groupby_features
    
def calculate_feature_click_count_within_time_range(groups, dt_backward, dt_forward, new_feature_name):
    counts_series = pd.Series([], name=new_feature_name)
    for one_group_key, one_group in groups: # one_group should be a Series.
        print(one_group_key)
        print(one_group)
        index_one_group = one_group.index # All index in current group.
        t_array = one_group.dt.to_pydatetime() # All timestamps in current group.
        counts_in_one_group_list = []
        for t in t_array:
            dt = t_array - t
            is_in_time_range = (dt >= dt_backward) & (dt <= dt_forward)
            counts_in_time_range = is_in_time_range.sum()
            counts_in_one_group_list.append(counts_in_time_range)
        # When the loop within one group is finished, append the result to counts_series.
        counts_series = counts_series.append(pd.Series(counts_in_one_group_list, index=index_one_group, name=new_feature_name))
    return counts_series

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
        
    def power_set(self, min_n=1, max_n=None):
        return tuple(self.iter_power_set(min_n=min_n, max_n=max_n))

class FeatureGenerator():
    @staticmethod
    def add_date_time(df:pd.DataFrame, time_column_name:str):
        print('\nAdding date time to dataframe...')
        #df['datetime'] = pd.to_datetime(df[time_column_name],format='%Y-%m-%d %H:%M:%S')
        df['year'] = df[time_column_name].dt.year.astype('int16')
        df['month'] = df[time_column_name].dt.month.astype('int8')
        df['weekday'] = df[time_column_name].dt.weekday.astype('int8') # The day of the week with Monday=0, Sunday=6.
        df['day'] = df[time_column_name].dt.day.astype('int8')
        df['hour'] = df[time_column_name].dt.hour.astype('int8')
        df['minute'] = df[time_column_name].dt.minute.astype('int8')
        df['second'] = df[time_column_name].dt.second.astype('int8')
        df['hour_of_day'] = (df['hour'] + df['minute'] / 60.0 + df['second'] / 3600.0).astype('float32')
        df['hour_of_day_sin'] = (df['hour_of_day'].apply(lambda x: math.sin((x - 12) / 24 * 2 * math.pi))).astype('float32')
        df['hour_of_day_cos'] = (df['hour_of_day'].apply(lambda x: math.cos((x - 12) / 24 * 2 * math.pi))).astype('float32')
        return df

    @staticmethod
    def add_click_time_delta_under_given_feature_combination(df:pd.DataFrame, time_column_name, features_iterable, direction=None, dtype='float32'):
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
                print('Adding feature: {}'.format(new_feature_name))
                df[new_feature_name] = (df[all_features].groupby(groupby_features)[time_column_name].shift(-1) - df[time_column_name]).dt.seconds.astype(dtype)
            elif direction == 'backward':
                new_feature_name = '_'.join(groupby_features) + '_dt_backward'
                print('Adding feature: {}'.format(new_feature_name))
                df[new_feature_name] = (df[time_column_name] - df[all_features].groupby(groupby_features)[time_column_name].shift(1)).dt.seconds.astype(dtype)
            elif direction == 'both':
                new_feature_name_forward = '_'.join(groupby_features) + '_dt_forward'
                new_feature_name_backward = '_'.join(groupby_features) + '_dt_backward'
                print('Adding feature: {}'.format(new_feature_name_forward))
                print('Adding feature: {}'.format(new_feature_name_backward))
                df[new_feature_name_forward] = (df[all_features].groupby(groupby_features)[time_column_name].shift(-1) - df[time_column_name]).dt.seconds.astype(dtype)
                df[new_feature_name_backward] = (df[time_column_name] - df[all_features].groupby(groupby_features)[time_column_name].shift(1)).dt.seconds.astype(dtype)
        return df

    @staticmethod
    def add_click_count_under_given_feature_combination(df:pd.DataFrame, features_iterable):
        print('\nAdding the count of each groups grouped by given feature combinations...')
        for groupby_features in features_iterable:
            groupby_features = list(groupby_features)
            new_feature_name = 'count_groupby_' +  '_'.join(groupby_features)
            print('\nAdding feature: {}'.format(new_feature_name))
            feature_click_count = df[groupby_features].groupby(by=groupby_features).size().reset_index().rename(columns={0:new_feature_name})
            # Merge feature_click_count using database-style join operation.
            df = df.merge(feature_click_count, on=groupby_features, how='left')
            # Clean memory.
            del feature_click_count # Removes a reference, which decrements the reference count on the value. 
            gc.collect() # Free memory of the unreferenced (reference count = 0) value.
        return df
    
    @staticmethod
    def add_click_count_under_given_feature_combination_within_time_range(df:pd.DataFrame, time_column_name,  features_iterable, dt_range=[]): # dt_range = [dt_backward, dt_forward](in seconds)
        assert isinstance(dt_range, list), 'dt_range must be a list!'
        # Check if dt_range is specified in argument list.
        if not dt_range: # dt_range is not specified. Calculate the total click count.
            print('\nAdding the count of each groups grouped by given feature combinations...')
            
            # Check how many cores on cpu.
            num_cpus = multiprocessing.cpu_count()
            print('\nTotol CPU cores on this node = {}'.format(num_cpus))
            # Create a multiprocessing pool with safe_lock.
            pool = multiprocessing.Pool(processes=num_cpus)
            # Create a list to save result.
            multiprocessing_results = []
            
            # Run calculate_feature_click_count() on every feature combinations.
            for groupby_features in features_iterable:
                # Return value: dictionary {new_feature_name: feature_click_count_df}
                result_temp = pool.apply_async(calculate_feature_click_count, args=(groupby_features,))
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
                feature_click_count_df, groupby_features = one_result.get()
                # Merge feature_click_count using database-style join operation.
                df = df.merge(feature_click_count_df, on=groupby_features, how='left')
            print('\nDone.')
            
            # Clean memory.
            del multiprocessing_results
            del feature_click_count_df
            gc.collect()
            
        else: # dt_range is specified.
            assert len(dt_range)==2, 'dt_range must be a 2 elements list!'
            dt_backward_seconds_int, dt_forward_seconds_int = int(dt_range[0]), int(dt_range[1])
            assert (dt_backward_seconds_int <= 0), 'The first element in dt_range must be <= 0!'
            assert (dt_forward_seconds_int >= 0), 'The second element in dt_range must be >= 0!'
            
            # Convert time shift to np.timedelta64.
            dt_backward_seconds = np.timedelta64(dt_backward_seconds_int, 's')
            dt_forward_seconds = np.timedelta64(dt_forward_seconds_int, 's')
            
            #print('\nAdding the count of each groups grouped by given feature combinations in dt range [{}, {}]...'.format(dt_backward_seconds, dt_forward_seconds))
            
            # Check how many cores on cpu.
            num_cpus = multiprocessing.cpu_count()
            print('\nTotol CPU cores on this node = {}'.format(num_cpus))
            # Create a multiprocessing pool with safe_lock.
            pool = multiprocessing.Pool(processes=num_cpus)
            
            for groupby_features in features_iterable:
                # Create a list to save result.
                multiprocessing_results = []
                all_indices_list = []
                all_counts_list = []
                groupby_features = list(groupby_features)
                new_feature_name = 'count_groupby_' +  '_'.join(groupby_features) + '_in_time_range_' + str(dt_backward_seconds_int) + '_' + str(dt_forward_seconds_int) + '_seconds'
                print('\nAdding feature: {}'.format(new_feature_name))
                
                # Group by groupby_features, then only keep the click_time column. 
                groups = df[groupby_features+[time_column_name]].groupby(by=groupby_features)[time_column_name]
                # Loop all group in groups to calculate counts within time range in each group.
                for name, group in groups: # group should be a Series.
                    #t_array = group.values.astype('datetime64[s]').view('int64')
                    #result_temp = pool.apply_async(check_time_distance_jit, args=(t_array, dt_backward_seconds_int, dt_forward_seconds_int,)) # Very Slow.
                    result_temp = pool.apply_async(check_time_distance, args=(group.values, dt_backward_seconds, dt_forward_seconds,))
                    multiprocessing_results.append(result_temp)
                    #print('index before:', len(all_indices_list))
                    all_indices_list += group.index.tolist() # Keep records of corresponding index.
                    #print('index after :', len(all_indices_list))
                    #print('index added :', len(group.index.tolist()))
                    #print('\n')
                    #print(group.index.tolist())
                # Wait for all worker processes in pool to finish their work.
                while len(pool._cache)!=0:
                    print('\n{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
                    time.sleep(1)
                print('\n{} - Done. There is 0 worker processes in pool.'.format(time.ctime()))                
                
                # When all worker processes are done with their work, unpack result.
                for one_result in multiprocessing_results:
                    #print('all_counts_list before:', len(all_counts_list))
                    all_counts_list += one_result.get()
                    #print('all_counts_list after :', len(all_counts_list))
                    #print('all_counts_list added :', len(one_result.get()))
                    #print('\n')           
                
                # Construct a dataframe with all_indices_list and all_counts_list.
                print(len(all_counts_list), len(all_indices_list))
                
                counts_in_time_range_df = pd.DataFrame(all_counts_list, index=all_indices_list, columns=[new_feature_name])
                
                # Merge according to index.
                df = df.merge(counts_in_time_range_df, left_index=True, right_index=True, how='left')
                
                # Clean memory.
                del multiprocessing_results
                del all_indices_list
                del all_counts_list
                del counts_in_time_range_df
                gc.collect()           
            
            # Close pool, prevent new worker process from joining.
            pool.close()
            # Block caller process until workder processes terminate.
            pool.join()
    
        return df
    
    @staticmethod
    def add_unique_count_of_feature_under_given_feature_combination(df:pd.DataFrame, features_iterable, objective_features:list):
        print('\nAdding the unique count of feature under given feature combinations...')
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
            feature_click_count_unique = df[all_features].groupby(by=groupby_features)[objective_features_without_groupby_features].nunique().reset_index().rename(columns=rename_dict)
            # Merge click_count using database-style join operation.
            df = df.merge(feature_click_count_unique, on=groupby_features, how='left')
            # Clean memory.
            del feature_click_count_unique
            gc.collect()
        return df


    @staticmethod
    def add_click_count_within_time_range_under_given_feature_combination(df:pd.DataFrame, features_iterable):
        pass
        
df_train_original = pd.read_feather('./data/input/train.feather.fraction_0.01')
df_train = df_train_original#.loc[:100000]
feature_to_combine_list = ['ip', 'app', 'device', 'os', 'channel']
#feature_to_combine_list = ['app', 'device']
feature_combine = FeatureCombination(feature_to_combine_list)
combined_feature = feature_combine.power_set()

df = FeatureGenerator.add_date_time(df_train, 'click_time')
#df = FeatureGenerator.add_click_time_delta_under_given_feature_combination(df, 'click_time', combined_feature, direction='both', dtype='float32')
#df = FeatureGenerator.add_unique_count_of_feature_under_given_feature_combination(df, combined_feature, feature_to_combine_list)
print(df.shape)

t0 = datetime.now()
df = FeatureGenerator.add_click_count_under_given_feature_combination_within_time_range(df, 'click_time',  combined_feature, dt_range=[0, 3600*10]) # dt_range in second.
t1 = datetime.now()

print((t1-t0).total_seconds())
print(df.shape)



'''df_train = pd.read_feather('./data/input/train.feather.small')
df2 = FeatureGenerator.add_date_time(df_train, 'click_time')
df2 = FeatureGenerator.add_click_time_delta_under_given_feature_combination(df2, 'click_time', combined_feature, direction='both', dtype='float32')
t0 = datetime.now()
df2 = FeatureGenerator.add_click_count_under_given_feature_combination(df2, combined_feature)#, dt_range=[-60, 60])
t1 = datetime.now()
print('multiprocessing:')
print((t1-t0).total_seconds())
print(df2.shape)
print(df.equals(df2))'''
    
    
    
    
    
    
