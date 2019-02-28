# Load all calcuated features, keep test_supplement_partial_processed index range.

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

# Load feature config file.
with open(os.path.join(input_dir,'feature_config.json'),'r',encoding='utf-8') as data_file:
    feature_dict = json.load(data_file)

# Flatten feature_dict to feature_list.
feature_list = []
for k, v in feature_dict.items():
    feature_list.extend(v)
    
# Check the existence of all feature files.
for feature in feature_list:
    feature_file_name = os.path.join(feature_dir, feature + '.feather')
    if os.path.isfile(feature_file_name):
        print('\nOK! {}.feather exists!'.format(feature))
    else:
        print('\nError! {}.feather does not exist!'.format(feature))
        sys.exit()

# Load test_supplement_partial_processed.
df_test_supplement_partial_processed = pd.read_feather('../data/input/test_supplement_partial_processed.feather')
# Load new test set.
df_test = pd.read_feather('../data/input/test.feather')

# Create mapping from new to old click id.
df_id_mapping = df_test.merge(df_test_supplement_partial_processed, on=list(df_test.columns.drop(['click_id'])), how='left')[['click_id_x', 'click_id_y']].drop_duplicates(subset=['click_id_x'], keep='first').rename(columns={'click_id_x': 'new_click_id', 'click_id_y': 'old_click_id'}).reset_index(drop=True)
selected_old_click_id = df_id_mapping['old_click_id'].values

# Use index of old click id to slice each features in test set.
for feature in feature_list:
    print('\n{} - Separating test set from feature:  {}...'.format(datetime.now().strftime(fmt), feature))
    df_test_feature = pd.read_feather(os.path.join(feature_dir, feature + '.feather')).iloc[-57536872:]
    df_test_feature.index = df_test_supplement_partial_processed['click_id'].values
    df_test_feature_selected_tmp = df_test_feature.loc[selected_old_click_id].reset_index(drop=True)
    print('\n{} - Saving test set feature:  {}...'.format(datetime.now().strftime(fmt), feature))
    df_test_feature_selected_tmp.to_feather(os.path.join(input_dir, 'test_' + feature + '.feather'))
    del df_test_feature_selected_tmp
    print('\n{} - Objects collected:  {}.\n'.format(datetime.now().strftime(fmt), gc.collect()))

# Combine click_id and all test set features.
df_test_generated = df_test['click_id']
for feature in feature_list:
    print('\n{} - Combining test set feature:  {}...'.format(datetime.now().strftime(fmt), feature))
    df_test_tmp = pd.read_feather(os.path.join(input_dir, 'test_' + feature + '.feather'))
    df_test_generated = pd.concat([df_test_generated, df_test_tmp], axis=1)
    print('\n{} - df_test_generated.shape = {}'.format(datetime.now().strftime(fmt), df_test_generated.shape))
    del df_test_tmp
    print('\n{} - Objects collected:  {}.\n'.format(datetime.now().strftime(fmt), gc.collect()))
    
print('\n{} - Saving combined test set...'.format(datetime.now().strftime(fmt)))
df_test_generated.to_feather(os.path.join(input_dir, 'test_generated_feature.feather'))
print('\n{} - Done.'.format(datetime.now().strftime(fmt)))


    
