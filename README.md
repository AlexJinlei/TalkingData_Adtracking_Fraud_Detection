# Kaggle TalkingData Adtracking Fraud Detection

## 1. Introduction
Kaggle competition page:
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## 2. Data Preprocessing
### 1_cvs_to_feather.py
Convert cvs data to feather format.
### 2_preprocess_feather_data.py
This code preprocesses the testing and training data. A new test data set was released by the organizer during the competition, so we combined the two set of test data, test.csv and supplement.csv. We add a new column to represent the time in int, which will be used later for feature generation. 
## 3. Model
10 bagged GBMs. Will add more details later.

## 3. Results
Late submission, ROC = 0.9828235, rank 24/3951
