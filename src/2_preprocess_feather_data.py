import os
import pandas as pd
from datetime import datetime

data_dir = '../data/input'

# 1) Loading the original data sets.
print('Loading "train.feather" ...')
t0 = datetime.now()
df_train = pd.read_feather(os.path.join(data_dir, 'train.feather'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_train.shape = {}\n'.format(df_train.shape))

'''
print('Loading "test.feather" ...')
t0 = datetime.now()
df_test = pd.read_feather(os.path.join(data_dir, 'test.feather'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_test.shape = {}\n'.format(df_test.shape))
'''

print('Loading "test_supplement.feather" ...')
t0 = datetime.now()
df_test_supplement = pd.read_feather(os.path.join(data_dir,  'test_supplement.feather'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_test_supplement.shape = {}\n'.format(df_test_supplement.shape))

#2) Add one more column to represent time in int. When represent 'click_time' with 'datetime64[s]' instead of 'datetime64[ns]', the max int view of 'click_time' is 1510329600. Since the max value of int32 is 2**31-1 = 2147483647 (check with np.iinfo(np.int32).max), we can safely store it as int32.
print('Adding "click_time_int_seconds" column...')
t0 = datetime.now()
df_train['click_time_int_seconds'] = df_train['click_time'].values.astype('datetime64[s]').view('int64').astype('int32')
df_test_supplement['click_time_int_seconds'] = df_test_supplement['click_time'].values.astype('datetime64[s]').view('int64').astype('int32')
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_train.shape = {}\n'.format(df_train.shape))
print('df_test_supplement.shape = {}\n'.format(df_test_supplement.shape))

# 3) Build data sets which will be used for feature generation.
# Time range:
# df_train           : 2017-11-06 14:32:21 ~ 2017-11-09 16:00:00
# df_test_supplement : 2017-11-09 14:23:39 ~ 2017-11-10 16:00:00
# df_test            : 2017-11-10 04:00:00 ~ 2017-11-10 15:00:00
# We will use both df_train and df_test_supplement to construct features. There is a small time range overlap between these two sets. According to our test, there are 1010 records in df_test_supplement in time range 2017-11-09 14:23:39 ~ 2017-11-09 16:00:00. We are going to drop these 1010 records when concatinate df_train and df_test_supplement, in order to keep the sorted click_time in combined data set.
# To reduce the disk space and memory usage, we are not going to really concatinate df_train and df_test_supplement. Instead, we are going to get a slice of df_test_supplement in which the records are in time range 2017-11-09 16:00:00 ~ 2017-11-10 16:00:00, and save it as a separated data set. We will append this set to df_train when we generate features.

# Get partial data set out of df_test_supplement.
print('Getting partial data set out of df_test_supplement...')
t0 = datetime.now()
df_test_supplement_partial = df_test_supplement[df_test_supplement['click_time'] >= df_train['click_time'].iloc[-1]]
df_test_supplement_partial.reset_index(drop=True, inplace=True)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))

# Save precessed data set to feather file.
print('Save precessed data set to feather file...')
t0 = datetime.now()
df_test_supplement_partial.to_feather(os.path.join(data_dir, 'test_supplement_partial_processed.feather'))
df_train.to_feather(os.path.join(data_dir, 'train_processed.feather'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))

# Down sampling for model developing. When model is established, use full data set.
fraction = 0.001
random_state = 114514
print('Down sampling for model developing...')
t0 = datetime.now()
df_train_small = df_train.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
df_test_supplement_partial_small = df_test_supplement_partial.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))

# Save downsampled data to feather file.
print('Saving downsampled data to feather file...')
t0 = datetime.now()
df_train_small.to_feather(os.path.join(data_dir, 'train_processed.feather.small'))
df_test_supplement_partial_small.to_feather(os.path.join(data_dir, 'test_supplement_partial_processed.feather.small'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))




