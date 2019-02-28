import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

input_dir = '../data/input/'
config = 'feature_list.json'
fmt = '%Y-%m-%d %H:%M:%S' # Time string format.

'''
# Load feature config file.
with open(os.path.join(input_dir,'feature_config.json'),'r',encoding='utf-8') as data_file:
    feature_dict = json.load(data_file)
'''

# Load raw features.
print('\nLoading processed train and test_supplement data...')
t0 = datetime.now()
df_train = pd.read_feather('../data/input/train_processed.feather')
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))

# Get the length of train data set. Will be used to split data.
len_train = df_train.shape[0]

# Split positive and negative samples.
df_train_positive = df_train.loc[df_train['is_attributed']==1]
df_train_negative = df_train.loc[df_train['is_attributed']==0]
# Get only index.
df_train_positive_index = pd.DataFrame(df_train_positive.index.tolist(), columns=['index'])
df_train_negative_index = pd.DataFrame(df_train_negative.index.tolist(), columns=['index'])

# Get the total number of positive samples.
n_positive = df_train_positive_index.shape[0]

dict_train_balanced_index_10_fold = {}
# Get equal amount of samples from negative samples. Sample 5 times.
for random_state_int in range(10):
    print('\n{} - Generating {} fold sample...'.format(datetime.now().strftime(fmt), random_state_int))
    df_train_negative_index_downsampled = df_train_negative_index.sample(n=n_positive, random_state=random_state_int)
    # Combine positive and downsampled negative samples. DO NOT reset index.
    df_train_balanced_index = pd.concat([df_train_positive_index, df_train_negative_index_downsampled], join='outer', axis=0)
    df_train_balanced_index.sort_values(inplace=True, by=['index'])
    # Add to dictionary.
    dict_train_balanced_index_10_fold['rand' + str(random_state_int)] = df_train_balanced_index['index'].values
print('\n{} - Done.'.format(datetime.now().strftime(fmt)))
# Convert dict to df.
df_train_balanced_index_10_fold = pd.DataFrame(dict_train_balanced_index_10_fold)
print(df_train_balanced_index_10_fold)
# Save balanced data.
df_train_balanced_index_10_fold.to_feather('../data/input/df_train_balanced_index_10_fold.feather')















