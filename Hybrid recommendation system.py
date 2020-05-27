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
test_file = sys.argv[2]
output_file = sys.argv[3]

sc=SparkContext()
sc.setLogLevel("ERROR")

#################################### item based functions ###########################################
def generate(frequent,size):
    result = []
    if(size!=2):
        for x in itertools.combinations(frequent,2):
            if len((x[0]).intersection(x[1])) >= size-2:
                new_set = ((x[0]).union(x[1]))
                if(new_set not in result):
                    subsets = itertools.combinations(new_set,size-1)
                    set(subsets)
                    if(set(subsets).issubset(frequent)):
                        result.append(new_set)
    else:
        for x in itertools.combinations(frequent, 2):
            result.append(((x[0]).union(x[1])))
    return result

def verify(frequent,result,size):
    candidates = []
    for x in result:
        flag = 0
        for y in itertools.combinations(x,size-1):
            if(frozenset(y) not in frequent):
                flag = 1
        if (flag ==0):
            candidates.append(x)
    return candidates

def count_and_filter(baskets, candidates, support):
    counts ={}
    result=[]
    for item in candidates:
        item_count = 0
        for basket in baskets:
            if (item).issubset(basket):
                if(item in counts.keys()):
                    counts[item] += 1
                else:
                    counts[item] = 1
    result = [item for (item, count) in counts.items() if count >= support]
    return result

def get_candidate_counts(user_business, candidate_itemsets):
    item_counts = defaultdict(int)
    for business in user_business:
        for item in candidate_itemsets:
            if set(business).issuperset(item):
                item_counts[item] += 1
    for item in sorted(item_counts.keys()):
        yield (item, item_counts[item])

############Functions for pearson formula ###############
def default_voting(current_ratings, total_ratings, size):
    while(size<50):
        current_ratings.append(3.7)
        total_ratings.append(3.7)
        size = size+1
    return current_ratings, total_ratings

def no_common_user(current_average_rating, total_average_rating):
    ############# This function will be called when the particlar current item has no co-related users #############
    ############# if there are no common user then I am using the ratio of current item to other total items ##############
    average_current = sum(current_average_rating)/len(current_average_rating)
    average_total = sum(total_average_rating)/len(total_average_rating)
    weight1 = float(average_current)/average_total
    weight2 = float(average_total)/average_current
    ############# we are returning min since we want weight <= 1 ##############
    return min(weight1,weight2)

def calculate_rating_dict(current_business, total_business, common_users):
    ############ creating a dictionary of ratings for current item and it's co-related items ##############
    current_ratings = [business_userRating[current_business][user] for user in common_users]
    total_ratings = [business_userRating[total_business][user] for user in common_users]
    return current_ratings, total_ratings

def calculate_rating_average(current_average_rating, total_average_rating):
    ############ calculating averges for the current item and it's co-related items ##############
    average_current = sum(current_average_rating) / float(len(current_average_rating))
    average_total = sum(total_average_rating) / float(len(total_average_rating))
    return average_current,average_total

def pearson_formula(current_ratings, total_ratings, average_current, average_total):
    ############# calculating the numerator and denominator for pearson formula ##############
    ############# p = (x1 - x1avg)(x2 - x2avg).../ sqrt(x1....)* sqrt(x2....) ##############
    current_diff = [val - average_current for val in current_ratings]
    total_diff = [val - average_total for val in total_ratings]

    current_denominator = sum([val * val for val in current_diff])** 0.5
    total_denominator = sum([val * val for val in total_diff]) ** 0.5

    numerator = sum([current_diff[val] * total_diff[val] for val in range(len(total_diff))])
    return numerator, current_denominator, total_denominator

def calculate_pearson_coefficient(iter):
    ############# this function is called to assign weights to using the pearson coefficient ##############
    ############# after assigning weights to the user later it will be used to pedict the rating for that item ##############
    user_id = iter[0]
    current_business = iter[1]
    remaining_business = iter[2]
    weights_dict = []
    current_users = business_users[current_business]
    current_average_rating = [business_userRating[current_business][user] for user in current_users]
    #current_average_rating = business_rating_broadcast.value[current_business]
    for i in remaining_business:
        total_users_i = business_users[i]
        common_users = total_users_i.intersection(current_users)
        total_average_rating = [business_userRating[i][user] for user in total_users_i]
        #total_average_rating = business_rating_broadcast.value[i]
        if len(common_users) == 0:
            weight = no_common_user(current_average_rating, total_average_rating)
            weights_dict.append([i, weight, business_userRating[i][user_id]])
            continue

        ############# if common_users is <50 we need to add default voting ##############
        # size = len(common_users)
        # current_ratings,total_ratings = default_voting(current_ratings,total_ratings, size)
        current_ratings,total_ratings = calculate_rating_dict(current_business, i , common_users)
        average_current,average_total = calculate_rating_average(current_average_rating, total_average_rating)
        numerator, current_denominator, total_denominator = pearson_formula(
            current_ratings, total_ratings, average_current, average_total)
        
        weight=0

        ############# edge cases ##############
        if(numerator==0 and (current_denominator==0 or total_denominator==0)):
            weight=1
        else:
            weight = numerator / (current_denominator * total_denominator)

        weights_dict.append([i, weight, business_userRating[i][user_id]])

    sorted_weights = sorted(weights_dict, key=lambda x: -x[1])

    return [(user_id, current_business), sorted_weights[:25]]

def pearson_rating_predict(items):
    ############# this function will be called when I need to predict rating for the item by the user ##############
    ############# using the assigned weight from the pearson formula I am predicting the rating of the item ##############
    current_user = items[0][0]
    current_business = items[0][1]
    weights = items[1]
    
    numerator = 0
    denominator = 0
    
    for w in weights:
        if(w[1]>0):
            numerator += w[1] * float(w[2])
            denominator += abs(w[1])
    if(denominator==0):
        pearson_prediction = 3.0
    if (denominator != 0):
        pearson_prediction = numerator / float(denominator)
        pearson_prediction = max(1, min(5, pearson_prediction))
    return [(current_user), {current_business: pearson_prediction}]

def cold_start(user, business, xgb_predicted_rating):
    ############# this function will be called when there is either new user or new business or both ##############
    ############# so basically, need to handle the above mentioned three cases separately ##############
    if user not in pearson_ratings and business not in pearson_ratings[user]:
        item_prediction = 3.0
        hybrid_prediction = 0.99 * xgb_predicted_rating + 0.01 * item_prediction
        file.write(x_test_collect[val][0] + "," + x_test_collect[val][1] + "," + str(hybrid_prediction) + "\n")
    elif business not in pearson_ratings[user]:
        val1=user_rating_broadcast.value[user]
        item_prediction = sum(val1)/float(len(val1))
        hybrid_prediction = 0.99 * xgb_predicted_rating + 0.01 * item_prediction
        file.write(x_test_collect[val][0] + "," + x_test_collect[val][1] + "," + str(hybrid_prediction) + "\n")
    else:
        val1 = business_rating_broadcast.value[business]
        item_prediction = sum(val1)/float(len(val1))
        hybrid_prediction = 0.99 * xgb_predicted_rating + 0.01 * item_prediction
        file.write(x_test_collect[val][0] + "," + x_test_collect[val][1] + "," + str(hybrid_prediction) + "\n")

#################################### model based functions ###########################################
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

text_rdd = sc.textFile(os.path.join(train_folder,'yelp_train.csv')).map(lambda line: line.split(","))
train_header = text_rdd.first()
text_rdd = text_rdd.filter(lambda row: row != train_header)

test_rdd = sc.textFile(test_file).map(lambda line: line.split(","))
test_header = test_rdd.first()
test_business = test_rdd.filter(lambda row: row != test_header).map(lambda row: (row[0], row[1]))

user_rating = text_rdd.map(lambda row: (row[0], {float(row[2])})).reduceByKey(lambda a, b: a | b)
user_rating = dict(user_rating.collect())
user_rating_broadcast = sc.broadcast(user_rating)

businesses_rdd = text_rdd.map(lambda row: (row[1], 1)).reduceByKey(lambda x, y: x).map(lambda x: x[0])
businesses = set(businesses_rdd.collect())

user_businesses = text_rdd.map(lambda row: (row[0], {row[1]})).reduceByKey(lambda a, b: a | b)
user_businesses = dict(user_businesses.collect())
############# using the user broadcast so that it can be using later as child instances ##############
user_businesses_broadcast = sc.broadcast(user_businesses)

business_users = text_rdd.map(lambda row: (row[1], {row[0]})).reduceByKey(lambda a, b: a | b)
business_users = dict(business_users.collect())

business_userRating = text_rdd.map(lambda row: (row[1], {row[0]: float(row[2])})).reduceByKey(lambda x, y: {**x, **y})
############# a dictionary consisting of business as it's key and user, their ratings for this business as value ##############
business_userRating = dict(business_userRating.collect())

test_business = test_business.filter(lambda x: x[1] in businesses)
############# it consist of all the business from train as well as test(validation) ##############
total_businesses = test_business.map(lambda row: (row[0], row[1], user_businesses_broadcast.value[row[0]]))

business_rating = text_rdd.map(lambda row: (row[1], {float(row[2])})).reduceByKey(lambda a, b: a | b)
business_rating = dict(business_rating.collect())
business_rating_broadcast = sc.broadcast(business_rating)

business_pearson_weight = total_businesses.map(calculate_pearson_coefficient)
pearson_predicted_ratings = business_pearson_weight.map(pearson_rating_predict).reduceByKey(lambda x, y: {**x, **y})
pearson_ratings = dict(pearson_predicted_ratings.collect())


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
    user = x_test_collect[val][0]
    business = x_test_collect[val][1]
    xgb_predicted_rating = check_prediction_rating(val)
    if user in pearson_ratings and business in pearson_ratings[user]:
        item_prediction = pearson_ratings[user][business]
        hybrid_prediction = 0.99 * xgb_predicted_rating + 0.01 * item_prediction
        file.write(x_test_collect[val][0] + "," + x_test_collect[val][1] + "," + str(hybrid_prediction) + "\n")
    else:
        cold_start(user, business, xgb_predicted_rating)
        
print(time.time() - start)