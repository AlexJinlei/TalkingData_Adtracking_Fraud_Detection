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
import multiprocessing

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


model_params = { \
'boosting_type': 'gbdt',\
'metric':['auc'], \
'objective':'binary', \
'learning_rate': 0.01,\
'max_depth': 7,\
'num_leaves':127, \
'subsample': 0.9,\
'subsample_freq': 1,\
'sub_feature': 0.4, \
'verbose': -1}
      
train_params = { \
'num_boost_round': 5000,\
'early_stopping_rounds': 100}

'''
for i in range(10):
    key = 'rand' + str(i)
    log('Training model using {} data...'.format(key))
    single_classifier = lgb.train(model_params, train_Dataset_dict[key]['train'], valid_sets=[train_Dataset_dict[key]['train'], train_Dataset_dict[key]['valid']], **train_params)
    log('Saving model {} to dictionary...'.format(key))
    model_dict[key] = single_classifier
    single_classifier.save_model(input_dir + 'single_lgbm_' + key + '.model', num_iteration=single_classifier.best_iteration)
'''

# Loading trained model.
bst = lgb.Booster(model_file='../input/single_lgbm_rand0.model')
y_pred = bst.predict(train_Dataset_dict[0]['valid']])

# Load test set.
test_generated_feature = pd.read_feather('../data/input/test_generated_feature.feather')
test_data = test_generated_feature[feature_list]
test_data_click_id = test_generated_feature['click_id']

pred_assemble = np.zeros(test_data.shape[0])
for i in range(10):
    key = 'rand' + str(i)
    log('Predict using classifier {}...'.format(key))
    single_classifier = model_dict[key]
    pred_single = single_classifier.predict(test_data, num_iteration=single_classifier.best_iteration)
    # Saving result to csv for submitting.
    log('Saving...')
    result_single = pd.DataFrame(pred_single, index=test_data_click_id)
    result_single = result_single.rename(columns={0:'is_attributed'})
    result_single.to_csv('./result_single_' + key + '_eta=0.01.csv')
    pred_assemble += pred_single

# Average 10 prediction results.
pred_assemble = pred_assemble/10.0
# Saving result to csv for submitting.
result_assemble = pd.DataFrame(pred_assemble, index=test_data_click_id)
result_assemble = result_assemble.rename(columns={0:'is_attributed'})
result_assemble.to_csv('./result_assemble_eta=0.01.csv')

'''
# Load csv and average 10 prediction results.
pred_assemble = np.zeros(test_data.shape[0])
for i in range(10):
    key = 'rand' + str(i)
    log('Loading result {}...'.format(key))
    df_pred_single = pd.read_csv('./result_single_' + key + '_eta=0.01.csv')
    pred_single = df_pred_single['is_attributed'].values
    pred_assemble += pred_single
'''    

'''
# !!! OverflowError: cannot serialize a bytes object larger than 4 GiB
# Predict with multiprocessing.
pool = multiprocessing.Pool(processes=10)
# Create a list to save result.
multiprocessing_results = []
# Run calculate_feature_click_count() on every feature combinations.
for i in range(10):
    key = 'rand' + str(i)
    log('Predict using classifier {}...'.format(key))
    single_classifier = model_dict[key]
    num_iteration = single_classifier.best_iteration
    result_temp = pool.apply_async(single_classifier.predict, args=(test_data,), kwds={'num_iteration':num_iteration})
    multiprocessing_results.append(result_temp)
# Close pool, prevent new worker process from joining.
pool.close()
# Block caller process until workder processes terminate.
pool.join()

# Unpack multiprocessing results.
pred_assemble = np.zeros(test_data.shape[0])
for one_result in multiprocessing_results:
    result_single = one_result.get()
    pred_assemble += pred_single
# Average 10 prediction results.
pred_assemble = pred_assemble/10.0

# Saving result to csv for submitting.
result_assemble = pd.DataFrame(pred_assemble, index=test_data_click_id)
result_assemble = result_assemble.rename(columns={0:'is_attributed'})
result_assemble.to_csv('./result_assemble_eta=0.01.csv')
'''

exit()

result_cv = lgb.cv(model_params, train_Dataset_dict['rand0']['all'], nfold=5, **train_params)
print('Best num_boost_round = {}'.format(len(result_cv['auc-mean'])))
print('Best CV score = {}'.format(max(result_cv['auc-mean'])))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



    
