# Load all calcuated features, keep test_supplement_partial_processed index range.

import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import sys
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def log(message):
    # fmt = '%Y-%m-%d %H:%M:%S' # Time string format.
    print('\n{} - {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))

input_dir = '../data/input/'
config = 'feature_list.json'


# Load feature config file.
with open(os.path.join(input_dir,'feature_config.json'),'r',encoding='utf-8') as data_file:
    feature_dict = json.load(data_file)

# Flatten feature_dict to feature_list.
feature_list = []
for k, v in feature_dict.items():
    feature_list.extend(v)

for removed_item in ['raw_channel', 'raw_ip', 'raw_device','raw_os']:
    feature_list.remove(removed_item)

categorical_feature = ['weekday', 'hour', 'minute'] # 'raw_channel', 'raw_ip', 'raw_app', 'raw_device', 'raw_os',

'''
train_Dataset_dict = {}
split_rand = 1234
test_size = 0.15
for i in range(10):
    key = 'rand' + str(i)
    log('Processing train and valid data {}...'.format(key))
    filename = key + '_train_and_label.feather'
    df_train_and_label_tmp = pd.read_feather(input_dir + filename)
    df_train_tmp = df_train_and_label_tmp[feature_list]
    df_label_tmp = df_train_and_label_tmp['is_attributed']
    X_train, X_valid, y_train, y_valid = train_test_split(df_train_tmp, df_label_tmp, test_size=test_size, random_state=split_rand)
    lgb_dataset_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)
    lgb_dataset_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)
    lgb_dataset_all = lgb.Dataset(df_train_tmp, label=df_label_tmp, feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)
    log('Saving...')
    lgb_dataset_train.save_binary(input_dir + key + '_lgb_train.bin')
    lgb_dataset_valid.save_binary(input_dir + key + '_lgb_valid.bin')
    lgb_dataset_all.save_binary(input_dir + key + '_lgb_all.bin')
    train_Dataset_dict[key] = {'train': lgb_dataset_train, 'valid': lgb_dataset_valid, 'all': lgb_dataset_all}

sys.exit()
'''



'''
# Load rand*_train_and_label.feather.
df_rand0_train_and_label = pd.read_feather('../data/input/rand0_train_and_label.feather')
# Load rand*_train_and_label.feather.
df_rand1_train_and_label = pd.read_feather('../data/input/rand1_train_and_label.feather')
# Load test set.
test_generated_feature = pd.read_feather('../data/input/test_generated_feature.feather')
test_data = test_generated_feature[feature_list]
test_data_click_id = test_generated_feature['click_id']


train_set = df_rand0_train_and_label[feature_list]
train_set_label = df_rand0_train_and_label['is_attributed']

train_set1 = df_rand1_train_and_label[feature_list]
train_set_label1 = df_rand1_train_and_label['is_attributed']

# Create train data for lgb model.
train_data_all = lgb.Dataset(train_set, label=train_set_label, feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)

train_data_all1 = lgb.Dataset(train_set1, label=train_set_label1, feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)

train_data = lgb.Dataset(train_set.iloc[:730953], label=train_set_label.iloc[:730953], feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)

valid_data = lgb.Dataset(train_set.iloc[730953:], label=train_set_label.iloc[730953:], feature_name=feature_list, categorical_feature=categorical_feature, free_raw_data=False)
'''

model_params = { \
'boosting_type': 'gbdt',\
'metric':['auc'], \
'objective':'binary', \
'learning_rate': 0.1,\
'max_depth': 7,\
'num_leaves':127, \
'subsample': 0.9,\
'subsample_freq': 1,\
'sub_feature': 0.4, \
'verbose': -1}
      
train_params = { \
'num_boost_round': 5000,\
'early_stopping_rounds': 100}

model_dict = {}

key = 'rand2'
bst = lgb.train(model_params, train_Dataset_dict[key]['train'], valid_sets=[train_Dataset_dict[key]['train'], train_Dataset_dict[key]['valid']], **train_params)
#roc_auc_score(train_set_label1, bst.predict(train_set1, num_iteration=bst.best_iteration))


pred_test_data = bst.predict(test_data)
result = pd.DataFrame(pred_test_data, index=test_data_click_id)
result = result.rename(columns={0:'is_attributed'})
result.to_csv('./result_submit4.csv')

exit()

result_cv = lgb.cv(model_params, train_Dataset_dict['rand0']['all'], nfold=5, **train_params)
print('Best num_boost_round = {}'.format(len(result_cv['auc-mean'])))
print('Best CV score = {}'.format(max(result_cv['auc-mean'])))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



    
