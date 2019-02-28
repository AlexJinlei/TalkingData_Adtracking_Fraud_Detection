import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import gc

# df_train.shape = (184903890, 9)
# test_supplement_partial_processed.shape = (57536872, 8)

input_dir = '../data/input/'
feature_dir = '../data/input/features/'
config = 'feature_list.json'
fmt = '%Y-%m-%d %H:%M:%S' # Time string format.

is_extract_label = 0

# Extract labels from original training set and save.
if is_extract_label:
    print('Loading "train.feather" ...')
    t0 = datetime.now()
    df_train = pd.read_feather(os.path.join(input_dir, 'train.feather'))
    t1 = datetime.now()
    print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
    print('df_train.shape = {}\n'.format(df_train.shape))
    # Extract label and save.
    df_train_label = df_train['is_attributed'].to_frame()
    df_train_label.to_feather(os.path.join(input_dir, 'df_train_label.feather'))

    # Extract downsampled label.
    df_train_balanced_index_10_fold = pd.read_feather(os.path.join(input_dir, 'df_train_balanced_index_10_fold.feather'))
    for sample_set_column in df_train_balanced_index_10_fold:
        sample_index = df_train_balanced_index_10_fold[sample_set_column]
        new_file_name = sample_set_column + '_train_label.feather'
        print('\n{} - Saving {}...'.format(datetime.now().strftime(fmt), new_file_name))
        df_train_label_tmp = df_train_label.loc[sample_index].reset_index().to_feather(os.path.join(input_dir, new_file_name))

# Load feature config file.
with open(os.path.join(input_dir,'feature_config.json'),'r',encoding='utf-8') as data_file:
    feature_dict = json.load(data_file)

# Flatten feature_dict to feature_list.
feature_list = []
for k, v in feature_dict.items():
    feature_list.extend(v)

# Generate rand0_train_set, rand1_train_set, ... rand9_train_set, with corresponding labels.
for i in range(10):
    prefix = 'rand' + str(i) + '_'
    print('\n{} - Generating {} train and label set...'.format(datetime.now().strftime(fmt), 'rand' + str(i)))
    # Check the existence of all feature files.
    for feature in feature_list:
        feature_file_name = os.path.join(input_dir, prefix + feature + '.feather')
        if os.path.isfile(feature_file_name):
            print('\nOK! {}.feather exists!'.format(prefix + feature))
        else:
            print('\nError! {}.feather does not exist!'.format(prefix + feature))
            sys.exit()
    # Create combined train and label dataframe.
    df_train_and_label_tmp = pd.DataFrame({})
    # Load label.
    df_label_tmp = pd.read_feather(os.path.join(input_dir, prefix + 'train_label.feather'))
    label_index_values = df_label_tmp['index'].values
    # Load downsampled feather.
    for feature in feature_list:
        print('\n{} - Combining feature {}...'.format(datetime.now().strftime(fmt), feature))
        df_feature = pd.read_feather(os.path.join(input_dir, prefix + feature + '.feather'))
        # Check the index of label and feather.
        if (df_feature['index'].values - label_index_values).sum():
            print('\n{} - Error! The indices of labels and training set are not same!'.format(datetime.now().strftime(fmt)))
            sys.exit()
        else:
            print('\n{} - OK! The indices of labels and training set are  same!'.format(datetime.now().strftime(fmt)))
        # Concatenate.
        df_train_and_label_tmp =  pd.concat([df_train_and_label_tmp, df_feature[feature]], axis=1)
        print('\n{} - Columns after combining: {}'.format(datetime.now().strftime(fmt), len(df_train_and_label_tmp.columns)))
    # After all feathers are combined, append label as the last column.
    df_train_and_label_tmp =  pd.concat([df_train_and_label_tmp, df_label_tmp['is_attributed']], axis=1)
    print('\n{} - Columns after combining: {}'.format(datetime.now().strftime(fmt), len(df_train_and_label_tmp.columns)))
    # Save.
    print('\n{} - Saving combined train and label file...'.format(datetime.now().strftime(fmt)))
    df_train_and_label_tmp.reset_index(drop=True).to_feather(os.path.join(input_dir, prefix + 'train_and_label.feather'))
    print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
    
    del df_label_tmp
    del label_index_values
    del df_feature
    del df_train_and_label_tmp
    gc.collect()
    
















