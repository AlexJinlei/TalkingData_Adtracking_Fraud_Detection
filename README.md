# Kaggle TalkingData Adtracking Fraud Detection

## 1. Introduction
Kaggle competition page:
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## 2. Data Preprocessing
### 1_cvs_to_feather.py
Convert cvs data to feather format.
### 2_preprocess_feather_data.py
This code preprocesses the testing and training data. A new test data set was released by the organizer during the competition, so we combined the two set of test data, test.csv and supplement.csv. We add a new column to represent the time in int, which will be used later for feature generation. 

## 3. Feature Engineer
Caution: feature engineer should be done to both training and testing data. We combined the training and testing data sets to one set to perform feature engineer. After feature engineering, we detach the training and testing data. There are five raw features in original training data, which are 'ip', 'app', 'device', 'os', and 'channel'. Firstly we created a power set of the raw features, in which the 'ip' feature must be included. Secondly, we add the date and time features to the data set, including 'year', 'month', 'weekday', 'day', 'hour', 'minute', 'second', 'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos'. There are 10 date and time features in total. The third step, we generate click time delta under given feature combination, including forward time delta and backward time delta. The fourth step, we create the unique count of feature under given feature combination. The last step is to generate click count under given feature combination within time range. We set the time range to 6 hours.

## 3. Model
10 bagged GBMs. Will add more details later.

## 3. Results
Late submission, ROC = 0.9828235, rank 24/3951
