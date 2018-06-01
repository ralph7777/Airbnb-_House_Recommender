# coding: utf-8

# # Airbnb Housing Reviews - User-based Recommender

# #### Group members: Kuangyi Zhang, Lanyixuan Xu, Jie Bao

# ## 1. Data Preprocessing

from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

#read the data
housing_reviews = pd.read_table("reviews.csv", header='infer', delimiter=",", na_values='NaN')

#drop the mmissing value
drop_rating_index = housing_reviews.index[housing_reviews['rating'] == 0]
drop_rating_index.tolist()
housing_reviews = housing_reviews.drop(housing_reviews.index[drop_rating_index])
housing_reviews = housing_reviews[~housing_reviews['comments'].isnull()] 
housing_reviews = housing_reviews[~housing_reviews['rating'].isnull()] 

#select the user with 3 or more comments for the house listing
recommend = housing_reviews[['listing_id', 'reviewer_id', 'rating']]
upper_2 = recommend['reviewer_id'].value_counts() > 3
users = upper_2[upper_2].index.tolist()
recommend = recommend[recommend['reviewer_id'].isin(users)]

#user_ id and hourse listing id 
list_id=recommend.iloc[0:,0].unique()
review_id = recommend.iloc[0:,1].unique()

#processing the data to build the matrix
dict = {}
for user in review_id:
    dict[user] = {}
number = 0
for i in range(len(recommend)):
    dict[recommend.iloc[i].reviewer_id][recommend.iloc[i].listing_id] = recommend.iloc[i].rating


recommend_list = []
for user in review_id:
    user_review = []
    for list in list_id:
        if list in dict[user].keys():
            user_review.append(dict[user][list])
        else:
            user_review.append(0)
    recommend_list.append(user_review)

dataMat =np.array(recommend_list)

def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item]>0, \
                                      dataMat[:,j]>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    #print 'standEst:', ratSimTotal, simTotal  
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    data=mat(dataMat)
    U,Sigma,VT = la.svd(data)
    Sig400 = mat(eye(400)*Sigma[:400]) #arrange Sig4 into a diagonal matrix
    xformedItems = data.T * U[:,:400] * Sig400.I  #create transformed items
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        #print 'svdEst:', similarity 
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating 
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def cross_validate_user(dataMat, user, test_ratio, estMethod=standEst, simMeas=ecludSim):
    number_of_items = np.shape(dataMat)[1]
    rated_items_by_user = np.array([i for i in range(number_of_items) if dataMat[user,i]>0])
    
    test_size = int(test_ratio * len(rated_items_by_user))
    test_indices = np.random.randint(0, len(rated_items_by_user), test_size)
    withheld_items = rated_items_by_user[test_indices]
    original_user_profile = np.copy(dataMat[user])
    dataMat[user, withheld_items] = 0 # So that the withheld test items is not used in the rating estimation below
    error_u = 0.0
    count_u = len(withheld_items)

    # Compute absolute error for user u over all test items
    for item in withheld_items:
        # Estimate rating on the withheld item
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        #print 'estimatedScore:', estimatedScore
        error_u = error_u + abs(estimatedScore - original_user_profile[item])
        #print error_u
        
    # Now restore ratings of the withheld items to the user profile
    for item in withheld_items:
        dataMat[user, item] = original_user_profile[item]

    # Return sum of absolute errors and the count of test cases for this user
    # Note that these will have to be accumulated for each user to compute MAE
    return error_u, count_u

def testMAE(dataMat, test_ratio, Method):
    total_error = 0
    total_count = 0
    for i in range(np.shape(dataMat)[0]):
        if Method == "standEst":
            error_u, count_u = cross_validate_user(dataMat, i, test_ratio, estMethod=standEst)
        elif Method == "svdEst":
            error_u, count_u = cross_validate_user(dataMat, i, test_ratio, estMethod=svdEst)
        total_error += error_u
        total_count += count_u
        #print error_u, count_u
    #print total_error, total_count
    print 'Mean Absoloute Error for', Method, ':', total_error/total_count


import operator
def most_similar_users(dataMat, userid, queryUser, k, metric=pearsSim):
    user  = dataMat[queryUser]
    sim = {}
    index = 0
    for i in dataMat:
        similarity = metric(i, user)
        if(similarity == 1):
            index = index + 1
            continue
        sim[index] = similarity
        index = index + 1
    sorted_keys = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
    index = 0
    for key, values in sorted_keys:
        if(index == k):
            break
        index += 1
    return sorted_keys

def predict(dataMat, user_id, user_index, list_index, similar_users, K):
    if len(similar_users) == 0:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    index = 0
    for key, values in similar_users:
        if(index == K):
            break
        if dataMat[key][list_index] != 0:
            neighbor_id = user_id[key]        
            neighbor_similarity = values
            rating = dataMat[key][list_index]
            numerator += neighbor_similarity * rating
            denominator += neighbor_similarity
            index += 1
    result = numerator/denominator
    return result 


def recommend_list(dataMat, user_id, list_id, queryUser, K):
    nopreference_list = np.where(dataMat[queryUser,:]==0)[0]
    predict_rating = {}
    sorted_most_similar_users = most_similar_users(dataMat, user_id, queryUser, K, metric=pearsSim)
    for item in nopreference_list:
        result = predict(dataMat, user_id, queryUser, item, sorted_most_similar_users, K)
        predict_rating[item] = result
    sorted_list = sorted(predict_rating.items(), key=operator.itemgetter(1), reverse=True)
    print ""
    print ""
    print '==================================='
    print '  Selected User:', user_id[queryUser]
    print '==================================='
    print 'The listing we recommend to this user: ','\n'

    index = 0
    for key, values in sorted_list:
        if(index == K):
            break
        print list_id[key], 'Website: https://www.airbnb.com/rooms/', list_id[key], '\n'
        print '------------------------------------------'
        index += 1

def recommender_user(list_id, review_id, dataMat):
	while True:
		print ""
		print ""
		print "Please enter you Airbnb user name. If your user name is our dataset, we will provide top several recommendations for you."
		print "Enter 'R' to back the menu"
		print ""
		print "REVIEW ID FOR TESTING:"
		print ""
		print review_id[0:20]
		intput = raw_input()
		user_index = 0
		if (intput == 'R' or intput == 'r'):
			break
		user_id = int(intput)
		if (user_id in review_id):
			user_index = np.where(user_id == review_id)[0][0]
			recommend_list(dataMat, review_id, list_id, user_index, 3)
		else:
			print "User name is not existed!"
			
def recommenderinterface(list_id, review_id, dataMat):
    while True:
    	print ""
    	print ""
    	print "==============================="
    	print "***** Recommender System ******"
    	print "==============================="
    	print ""
    	print "Two recommendation systems:"
    	print "---1) ITEM-BASED COLLABORATIVE RECOMMENDER"
    	print "---2) USER-BASED COLLABORATIVE RECOMMENDER"
    	print "One prediction rating system:"
    	print "---3) SUGGESTED RATING FRO USER"
    	print ""
    	print "Please type '1' or '2' or '3'"
    	print "Enter 'Q' to quit the application"
    	inp = raw_input()
    	if inp == '2':
    		recommender_user(list_id, review_id,dataMat)
    	if (inp == 'Q' or inp == 'q'):
			break

recommenderinterface(list_id, review_id, dataMat)
