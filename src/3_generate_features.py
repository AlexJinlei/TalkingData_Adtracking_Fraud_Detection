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
import sys
import os
import gc
from Features import *

if __name__ == '__main__':
    
    # Note: multiprocessing.set_start_method('forkserver') should be guarded with in __name__ == '__main__':
    multiprocessing.set_start_method('forkserver') # Child do not inherit unneccesary resources.

    output_dir = '../data/input/features/'

    is_test = 0

    is_output_log = 1

    if is_output_log:
        # Set log.
        log_bufsize = 1 # 0 means unbuffered, 1 means line buffered.
        log_filename = 'generate_features_log_' + '{:%Y%m%d_%H-%M-%S}'.format(datetime.now()) + '.txt'
        log_path = output_dir
        log_path_filename = log_path + log_filename
        log = open(log_path_filename, 'w', log_bufsize)
        sys.stdout = log

    # Loading preprocessed data sets.
    print('\nLoading processed train and test_supplement data...')
    t0 = datetime.now()
    if is_test:
        df_train = pd.read_feather('../data/input/train_processed.feather.small')
        df_test = pd.read_feather('../data/input/test_supplement_partial_processed.feather.small')
    else:
        df_train = pd.read_feather('../data/input/train_processed.feather')
        df_test = pd.read_feather('../data/input/test_supplement_partial_processed.feather')
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))

    # Get the length of each data set. Will be used to split data.
    len_train = df_train.shape[0]
    len_test = df_test.shape[0]

    # Combine train and test data set. Keep only common columns (use inner join).
    # df_train == df_all.iloc[0:len_train], df_test == df_all.iloc[len_train:len_train+len_test]
    print('\nConcatenating train and test data sets...')
    t0 = datetime.now()
    # The index of df_all must be reset. The following functions will use the index for dataframe merge.
    df_all = pd.concat([df_train, df_test], join='outer', sort=False).reset_index(drop=True)
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    
    '''
    # Split each raw features.
    for feature in df_all:
        print('Saving raw feature: {}...'.format(feature))
        df_all[feature].to_frame().to_feather(os.path.join(output_dir, 'raw_' + feature + '.feather'))
    '''
    
    
    # Specify the raw features which are used to generate new ones.
    feature_to_combine_list = ['ip', 'app', 'device', 'os', 'channel']

    # Generate power set of raw features.
    feature_combine = FeatureCombination(feature_to_combine_list)
    combined_feature = feature_combine.power_set()
    combined_feature_include_ip = feature_combine.power_set(include_list=['ip'])
    print(combined_feature)
    print('\n')
    print(combined_feature_include_ip)
    generated_features = []

    
    '''
    # Add date time features.
    print('\nGenerating date time features...')
    t0 = datetime.now()
    feature_added = FeatureGenerator.add_date_time(df_all, 'click_time', output_dir)
    generated_features += feature_added
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    print('feature_added = {}'.format(feature_added))
    '''
    

    
    print('\nGenerating click_time_delta_under_given_feature_combination...')
    t0 = datetime.now()
    feature_added = FeatureGenerator.add_click_time_delta_under_given_feature_combination(df_all, 'click_time', combined_feature, output_dir, direction='both', dtype='float32')
    generated_features += feature_added
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    print('feature_added = {}'.format(feature_added))
    

    '''
    print('\nGenerating unique_count_of_feature_under_given_feature_combination...')
    t0 = datetime.now()
    feature_added = FeatureGenerator.add_unique_count_of_feature_under_given_feature_combination(df_all, combined_feature, feature_to_combine_list, output_dir)
    generated_features += feature_added
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    print('feature_added = {}'.format(feature_added))
    '''
    
    '''
    print('\nGenerating click_count_under_given_feature_combination_within_time_range...')
    t0 = datetime.now()
    n_hours = 6
    feature_added = FeatureGenerator.add_click_count_under_given_feature_combination_within_time_range(df_all, 'click_time_int_seconds',  (combined_feature_include_ip[15],), output_dir, dt_range=[0, 3600*n_hours]) # dt_range in second.
    generated_features += feature_added
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    print('feature_added = {}'.format(feature_added))
    '''


    
    
    
    
    
    
