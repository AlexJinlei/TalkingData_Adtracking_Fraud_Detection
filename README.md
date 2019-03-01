# Kaggle TalkingData Adtracking Fraud Detection

## 1. Introduction
Kaggle competition page:
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## 2. Data Preprocessing
### Code: 1_cvs_to_feather.py
Convert cvs data to feather format.
### Code: 2_preprocess_feather_data.py
This code preprocesses the testing and training data. A new test data set was released by the organizer during the competition, so we combined the two set of test data, test.csv and supplement.csv. We add a new column to represent the time in int, which will be used later for feature generation. 

## 3. Feature Engineer
### Code: 3_generate_features.py
Caution: feature engineer should be done to both training and testing data. We combined the training and testing data sets to one set to perform feature engineer. After feature engineering, we detach the training and testing data. There are five raw features in original training data, which are 'ip', 'app', 'device', 'os', and 'channel'. Firstly we created a power set of the raw features, in which the 'ip' feature must be included. Secondly, we add the date and time features to the data set, including 'year', 'month', 'weekday', 'day', 'hour', 'minute', 'second', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos'. There are 10 date and time features in total, but finally we only use 'weekday', 'hour', 'minute', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos'. The third step, we generate click time delta under given feature combination, including forward time delta and backward time delta. The fourth step, we create the unique count of feature under given feature combination. The last step is to generate click count under given feature combination within time range. We set the time range to 1 hour and 6 hours.

After feature engineering, we have 5 sets of features, they are:

### raw
'raw_channel', 'raw_ip', 'raw_app', 'raw_device', 'raw_os'

### date_time
'weekday', 'hour', 'minute', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos'

### time_delta
'ip_dt_forward', 'ip_dt_backward', 'app_dt_forward', 'app_dt_backward', 'device_dt_forward', 'device_dt_backward', 'os_dt_forward', 'os_dt_backward', 'channel_dt_forward', 'channel_dt_backward', 'ip_app_dt_forward', 'ip_app_dt_backward', 'ip_device_dt_forward', 'ip_device_dt_backward', 'ip_os_dt_forward', 'ip_os_dt_backward', 'ip_channel_dt_forward', 'ip_channel_dt_backward', 'app_device_dt_forward', 'app_device_dt_backward', 'app_os_dt_forward', 'app_os_dt_backward', 'app_channel_dt_forward', 'app_channel_dt_backward', 'device_os_dt_forward', 'device_os_dt_backward', 'device_channel_dt_forward', 'device_channel_dt_backward', 'os_channel_dt_forward', 'os_channel_dt_backward', 'ip_app_device_dt_forward', 'ip_app_device_dt_backward', 'ip_app_os_dt_forward', 'ip_app_os_dt_backward', 'ip_app_channel_dt_forward', 'ip_app_channel_dt_backward', 'ip_device_os_dt_forward', 'ip_device_os_dt_backward', 'ip_device_channel_dt_forward', 'ip_device_channel_dt_backward', 'ip_os_channel_dt_forward', 'ip_os_channel_dt_backward', 'app_device_os_dt_forward', 'app_device_os_dt_backward', 'app_device_channel_dt_forward', 'app_device_channel_dt_backward', 'app_os_channel_dt_forward', 'app_os_channel_dt_backward', 'device_os_channel_dt_forward', 'device_os_channel_dt_backward', 'ip_app_device_os_dt_forward', 'ip_app_device_os_dt_backward', 'ip_app_device_channel_dt_forward', 'ip_app_device_channel_dt_backward', 'ip_app_os_channel_dt_forward', 'ip_app_os_channel_dt_backward', 'ip_device_os_channel_dt_forward', 'ip_device_os_channel_dt_backward', 'app_device_os_channel_dt_forward', 'app_device_os_channel_dt_backward', 'ip_app_device_os_channel_dt_forward', 'ip_app_device_os_channel_dt_backward'

### unique_count
'unique_count_app_groupby_ip', 'unique_count_channel_groupby_ip', 'unique_count_device_groupby_ip', 'unique_count_os_groupby_ip', 'unique_count_channel_groupby_app', 'unique_count_device_groupby_app', 'unique_count_ip_groupby_app', 'unique_count_os_groupby_app', 'unique_count_app_groupby_device', 'unique_count_channel_groupby_device', 'unique_count_ip_groupby_device', 'unique_count_os_groupby_device', 'unique_count_app_groupby_os', 'unique_count_channel_groupby_os', 'unique_count_device_groupby_os', 'unique_count_ip_groupby_os', 'unique_count_app_groupby_channel', 'unique_count_device_groupby_channel', 'unique_count_ip_groupby_channel', 'unique_count_os_groupby_channel', 'unique_count_channel_groupby_ip_app', 'unique_count_device_groupby_ip_app', 'unique_count_os_groupby_ip_app', 'unique_count_app_groupby_ip_device', 'unique_count_channel_groupby_ip_device', 'unique_count_os_groupby_ip_device', 'unique_count_app_groupby_ip_os', 'unique_count_channel_groupby_ip_os', 'unique_count_device_groupby_ip_os', 'unique_count_app_groupby_ip_channel', 'unique_count_device_groupby_ip_channel', 'unique_count_os_groupby_ip_channel', 'unique_count_channel_groupby_app_device', 'unique_count_ip_groupby_app_device', 'unique_count_os_groupby_app_device', 'unique_count_channel_groupby_app_os', 'unique_count_device_groupby_app_os', 'unique_count_ip_groupby_app_os', 'unique_count_device_groupby_app_channel', 'unique_count_ip_groupby_app_channel', 'unique_count_os_groupby_app_channel', 'unique_count_app_groupby_device_os', 'unique_count_channel_groupby_device_os', 'unique_count_ip_groupby_device_os', 'unique_count_app_groupby_device_channel', 'unique_count_ip_groupby_device_channel', 'unique_count_os_groupby_device_channel', 'unique_count_app_groupby_os_channel', 'unique_count_device_groupby_os_channel', 'unique_count_ip_groupby_os_channel', 'unique_count_channel_groupby_ip_app_device', 'unique_count_os_groupby_ip_app_device', 'unique_count_channel_groupby_ip_app_os', 'unique_count_device_groupby_ip_app_os', 'unique_count_device_groupby_ip_app_channel', 'unique_count_os_groupby_ip_app_channel', 'unique_count_app_groupby_ip_device_os', 'unique_count_channel_groupby_ip_device_os', 'unique_count_app_groupby_ip_device_channel', 'unique_count_os_groupby_ip_device_channel', 'unique_count_app_groupby_ip_os_channel', 'unique_count_device_groupby_ip_os_channel', 'unique_count_channel_groupby_app_device_os', 'unique_count_ip_groupby_app_device_os', 'unique_count_ip_groupby_app_device_channel', 'unique_count_os_groupby_app_device_channel', 'unique_count_device_groupby_app_os_channel', 'unique_count_ip_groupby_app_os_channel', 'unique_count_app_groupby_device_os_channel', 'unique_count_ip_groupby_device_os_channel', 'unique_count_channel_groupby_ip_app_device_os', 'unique_count_os_groupby_ip_app_device_channel', 'unique_count_device_groupby_ip_app_os_channel', 'unique_count_app_groupby_ip_device_os_channel', 'unique_count_ip_groupby_app_device_os_channel'

### count_groupby_0_3600
'count_groupby_ip_app_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_os_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_in_time_range_0_3600_seconds',
 'count_groupby_ip_device_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_os_channel_in_time_range_0_3600_seconds',
 'count_groupby_ip_app_device_os_channel_in_time_range_0_3600_seconds'
 
 ### count_groupby_0_21600
 'count_groupby_ip_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_in_time_range_0_21600_seconds',
 'count_groupby_ip_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_os_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_app_device_os_channel_in_time_range_0_21600_seconds',
 'count_groupby_ip_device_in_time_range_0_21600_seconds'
 
Feature counts under each category are:

#### raw: 5

#### date_time: 6

#### time_delta: 62

#### unique_count: 75

#### count_groupby_0_3600: 16

#### count_groupby_0_21600: 16

#### total: 180

## 4. Negative Downsample
### Code: 4_generate_negative_downsample_index.py, 5_negative_downsample_features.py
In training data set, the positive and negative records are extremely imbalanced (Positive/Negative = 0.25%). We adopted negative downsampled strategy to balance the positive and negative records. After downsampling, the positive and negative recordes are approximately equal.

### Code: 6_generate_training_set_and_label.py, 7_separate_test_set.py
After downsample, we seperate training set and testing set.

## 5. Model
We use gradient boosting decision tree to train our model. A simple GBM can achieve ROC score 0.98. Two further improve our score, we bagged 10 GBMs to make an assembled model. The assembled model achieved the ROC score 0.9828235. After fine tuning, we got the best parameters:

model_params = { 'boosting_type': 'gbdt', 'metric':['auc'], 'objective':'binary',  'learning_rate': 0.01, 'max_depth': 7,'num_leaves':127,  'subsample': 0.9, 'subsample_freq': 1, 'sub_feature': 0.4, 'verbose': -1}
      
train_params = { 'num_boost_round': 5000, 'early_stopping_rounds': 100}

## 6. Results
Late submission, ROC = 0.9828235, rank 24/3951
