# Energy-predictor
ASHRE-Great-Energy-Predictor-3

Project is implemented in python programming. Data cleaning and Model implementation is done using necessary packages and library. Evaluation metrics for this competition is Root Mean Squared Logarithmic Error (RMSLE). Pipeline and Methods: Data Information: • Data of project was in 6 files in Kaggle. We transformed data into Google Drive for its efficient working in Google Collaboratory. • We combined data from these files to get resultant file of train and test. This is using aggregate and group by function in pandas. • We then took input from test and train files for further processing. • As observed there were 16 columns in train data frame with 20216100 rows in train data set. • As observed there were 15 columns in train data frame with 41697600 rows in test data set.

Data Mining and Cleaning: Some Statistics: Total rows in train: 20216100 Total rows in test: 41697600

• On our analysis and discussion for given dataset, we have made necessary conversions by multiplying meter_reading (target variable) with 0.2931 for meter = 0

Referred link for the discussion is: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261

• It can also be observed the tuples with meter_reading=0 have very less significance in training and prediction. Therefore, we removed all the tuples with meter_reading=0 from train data frame.

• Next task was to check for missing values in features and analyze their Importance and performance in training and prediction.

• We plotted all features of the dataset to visualize count and variance of them across bins.

• We also observed variance and behavior of features based on months of the year. This gave us consisting pattern across months. We then decided to drop column “timestamp”.

•To deal with NaN values we first calculated the total percentage of NaN values for individual features in building and whether dataset. Based on the analysis, we then removed features with high percentage of NaN values as they seemed irrelevant for training the model. Columns ignored are: Year_built, Floor_count, precip_depth_1_hr, Cloud_coverage

•Column building_id, site_id, meter and timestamp were also ignored as their primary purpose was to aggregate data and they were having less impact in prediction.

•Final Dataset after cleaning and handling missing values were having following features:

square_feet air_temperature dew_temperature sea_level_pressure wind_direction wind_speed meter_reading

•We repeated the above cleaning process on test data to get above feature as a result (except meter_reading). •We divided the train data frame in training features and label namely X and y

Training Models: We tried following algorithms from our coursework to train the data set. RMSLE was initially calculated by prediction on train data itself. Upon successful result on train we predicted test labels i.e. meter_reading, and submitted them in Kaggle to get final RMSLE

1.Linear Regression: This model was not successful as it gave negative values in prediction which is logically incorrect and unacceptable for RMSLE.

2.K Nearest neighbor (KNN)Regressor: This model gave us the best RMSLE score.

Train Score: 1.7001 Test Score: 1.91

3.Decision Tree Regressor: Train Score: 1.67 Test Score: 1.94

4.Random Forest Regressor: As seen Random Forest performed better than Decision Tree due to its wide functionality of sampling features and data. Train Score: 1.6 Test Score: 1.93

5.LightGBM Gradient Boosting: This algorithm was suggested by many people in discussions. But we failed to get RMSLE on it. Train Score: 11

6.Adaboost: Train Score: 5 7.

7.Gradient Boost: Train Score: 2.79
