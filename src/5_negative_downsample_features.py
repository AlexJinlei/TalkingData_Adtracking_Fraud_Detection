import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import gc

# df_train.shape = (184903890, 9)
# test_supplement_partial_processed.shape = (57536872, 8)

input_dir = '../data/input/'
feature_dir = '../data/input/features/'
config = 'feature_list.json'
fmt = '%Y-%m-%d %H:%M:%S' # Time string format.

# Load feature config file.
with open(os.path.join(input_dir,'feature_config.json'),'r',encoding='utf-8') as data_file:
    feature_dict = json.load(data_file)

# Flatten feature_dict to feature_list.
feature_list = []
for k, v in feature_dict.items():
    feature_list.extend(v)

# Load Sample index file.
df_train_balanced_index_10_fold = pd.read_feather('../data/input/df_train_balanced_index_10_fold.feather')

# Check the existence of all feature files.

for feature in feature_list:
    if os.path.isfile(os.path.join(feature_dir, feature + '.feather')):
        print('\nOK! {}.feather exists!'.format(feature))
        feature_df = pd.read_feather(os.path.join(feature_dir, feature + '.feather'))
        column_name = feature_df.columns.tolist()[0]
        print('\nColumn name: {}'.format(column_name))
        if column_name != feature:
            feature_df.rename(columns={column_name:feature}).to_feather(os.path.join(feature_dir, feature + '.feather'))
            print('\nRename column name from {} to {}'.format(column_name, feature))
        # Downsample and save.
        for sample_set_column in df_train_balanced_index_10_fold:
            print('\n{} - Creating fold {}...'.format(datetime.now().strftime(fmt), sample_set_column))
            sample_index = df_train_balanced_index_10_fold[sample_set_column]
            new_file_name = sample_set_column + '_' + feature + '.feather'
            print('\n{} - Saving {}...'.format(datetime.now().strftime(fmt), new_file_name))
            feature_df.loc[sample_index].reset_index().to_feather(os.path.join(input_dir, new_file_name))
        # Release memory.
        del feature_df
        del sample_index
        print('\n{} - Objects collected: {}.'.format(datetime.now().strftime(fmt), gc.collect()))
    else:
        print('\nError! {}.feather does not exist!'.format(feature))



'''
# Loop resample index sets.
for sample_set_column in df_train_balanced_index_10_fold:
    print('\n{} - Creating fold {}...'.format(datetime.now().strftime(fmt), sample_set_column)) 
    sample_index = df_train_balanced_index_10_fold[sample_set_column]
    train_set_tmp = pd.DataFrame({})
    # Use sample_index to extract training set from full feature set.
    for feature in feature_list:
        print('\n{} - Sampling feature {}...'.format(datetime.now().strftime(fmt), feature))
        feature_df = pd.read_feather(os.path.join(feature_dir, feature + '.feather'))
        feature_sampled_df = feature_df.loc[sample_index]
        # Append column to train_set_tmp.
        train_set_tmp = pd.concat([train_set_tmp, feature_sampled_df], axis=1)
        del feature_df
        del feature_sampled_df
        gc.collect()
    print('\n{} - Saving fold {}...'.format(datetime.now().strftime(fmt), sample_set_column))
    train_set_tmp.reset_index().to_feather(os.path.join(input_dir, 'train_' + sample_set_column + '.feather'))
    del train_set_tmp
    gc.collect()
'''














