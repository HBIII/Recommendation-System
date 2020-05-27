import math
import os
import sys
import json
import numpy as np
import xgboost as xgb
from pyspark import SparkContext, SparkConf
import time

start = time.time()

train_folder = sys.argv[1]
x_test = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext()
sc.setLogLevel("ERROR")

def constructUserData():
    user = sc.textFile(os.path.join(train_folder, 'user.json')).map(json.loads)
    user = user.map(
        lambda x: (x["user_id"], (x["review_count"], x["average_stars"], x["useful"], x["fans"]))).collectAsMap()
    return user

def constructBusinessData():
    business = sc.textFile(os.path.join(train_folder, 'business.json')).map(json.loads)
    business = business.map(
        lambda x: (x['business_id'], (x['stars'], x['review_count']))).collectAsMap()
    return business

def X_train_mapping(data):
    mappings = []
    mappings.extend(userRDD_data.get(data[0]))
    mappings.extend(businessRDD_data.get(data[1]))
    return mappings

def X_test_mapping(data):
    mappings = []
    mappings.extend(userRDD_data.get(data[0]))
    mappings.extend(businessRDD_data.get(data[1]))
    return mappings

def check_prediction_rating(val):
    xgb_predicted_rating = xgb_prediction[val]
    xgb_predicted_rating = xgb_predicted_rating = 5.0 if xgb_predicted_rating > 5 else xgb_predicted_rating
    xgb_predicted_rating = xgb_predicted_rating = 1.0 if xgb_predicted_rating < 1 else xgb_predicted_rating
    return xgb_predicted_rating

train_rdd = sc.textFile(os.path.join(train_folder,'yelp_train.csv'))
train_header = train_rdd.first()
train_data_collect = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(",")).map(
    lambda x: (x[0], x[1], float(x[2]))).collect()

x_test = sc.textFile(test_file)
test_header = x_test.first()
x_test_filter = x_test.filter(lambda x: x != test_header)
x_test_collect = x_test_filter.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1])).collect()

userRDD_data = constructUserData()
businessRDD_data = constructBusinessData()

X_train = []
Y_train = []
for data in train_data_collect:
    mappings = X_train_mapping(data)
    X_train.append(mappings)
    Y_train.append(data[2])

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

X_test = []
for data in x_test_collect:
    mappings = X_test_mapping(data)
    X_test.append(mappings)

X_test = np.asarray(X_test)

xgb_model = xgb.XGBRegressor(max_depth=25, steps=30, gamma=20, learning_rate=0.3, n_estimators=150, booster = 'gbtree')
xgb_model.fit(X_train, Y_train)
xgb_prediction = xgb_model.predict(X_test)

file = open(output_file, 'w')
file.write("user_id, business_id, prediction\n")
for val in range(0, len(xgb_prediction)):
    xgb_predicted_rating = check_prediction_rating(val)
    file.write(x_test_collect[val][0] + "," + x_test_collect[val][1] + "," + str(xgb_predicted_rating) + "\n")

output_rdd = sc.textFile(output_file)
output_header = output_rdd.first()
output_data = output_rdd.filter(lambda x: x != output_header).map(lambda x: x.split(','))
output_data_dict = output_data.map(lambda x: (((x[0]), (x[1])), float(x[2])))
test_data_dict = x_test_filter.map(lambda x: x.split(",")).map(lambda x: (((x[0]), (x[1])), float(x[2])))
joined_data = test_data_dict.join(output_data_dict).map(lambda x: (abs(x[1][0] - x[1][1])))

rmse_rdd = joined_data.map(lambda x: x ** 2).reduce(lambda x, y: x + y)
rmse = math.sqrt(rmse_rdd / output_data_dict.count())
print("RMSE", rmse)

print("Duration : ", time.time() - start)