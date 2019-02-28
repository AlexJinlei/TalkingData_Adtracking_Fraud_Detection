import os
import pandas as pd
from datetime import datetime

data_dir = './data/input'

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

test_columns = ['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
          'ip': 'uint32',
          'app': 'uint16',
          'device': 'uint16',
          'os': 'uint16',
          'channel': 'uint16',
          'click_time': 'str',
          'attributed_time': 'str',
          'is_attributed': 'uint8',
          'click_id': 'uint32'}
          
train_parse_dates = ['click_time', 'attributed_time']
test_parse_dates = ['click_time']

print('Loading "train.csv" ...')
t0 = datetime.now()
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), dtype=dtypes, usecols=train_columns, parse_dates=train_parse_dates)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_train.shape = {}\n'.format(df_train.shape))

print('Loading "test.csv" ...')
t0 = datetime.now()
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'), dtype=dtypes, usecols=test_columns, parse_dates=test_parse_dates)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_test.shape = {}\n'.format(df_test.shape))

print('Loading "test_supplement.csv" ...')
t0 = datetime.now()
df_test_supplement = pd.read_csv(os.path.join(data_dir, 'test_supplement.csv'), dtype=dtypes, usecols=test_columns, parse_dates=test_parse_dates)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.'.format((t1-t0).total_seconds()/60))
print('df_test_supplement.shape = {}\n'.format(df_test_supplement.shape))

# Drop the records from supplement test set which are possibly contained in train set.
df_old_test = df_test_supplement[df_test_supplement['click_time'] >= df_train['click_time'].iloc[-1]]
df_old_test.reset_index(drop=True, inplace=True)

# Convert data to feather format.
print('Converting data file to feather format...')
t0 = datetime.now()
df_train.to_feather(os.path.join(data_dir, 'train.feather'))
df_test.to_feather(os.path.join(data_dir, 'test.feather'))
df_test_supplement.to_feather(os.path.join(data_dir, 'test_supplement.feather'))
df_old_test.to_feather(os.path.join(data_dir, 'old_test.feather'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))

# Down sampling for model developing. When model is established, use full data set.
fraction = 0.001
random_state = 114514
print('Down sampling for model developing...')
t0 = datetime.now()
df_train_small = df_train.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
df_test_small = df_test.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
df_old_test_small = df_old_test.sample(frac=fraction, random_state=random_state).sort_values(by='click_time').reset_index(drop=True)
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))

# Save downsampled data to feather file.
print('Saving downsampled data to feather file...')
t0 = datetime.now()
df_train_small.to_feather(os.path.join(data_dir, 'train.feather.small'))
df_test_small.to_feather(os.path.join(data_dir, 'test.feather.small'))
df_old_test_small.to_feather(os.path.join(data_dir, 'old_test.feather.small'))
t1 = datetime.now()
print('Done. Time elapsed: {:.2f} minutes.\n'.format((t1-t0).total_seconds()/60))

