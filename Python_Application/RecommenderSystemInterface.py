# coding: utf-8

# # Airbnb Housing Reviews - Item/User-based Recommender + SUGGESTED RATING FRO USER

# #### Group members: Kuangyi Zhang, Lanny (Lanyixuan) Xu, Jie Bao


import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)

from numpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import pickle
import warnings


import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

CountV = joblib.load('CountVectorizer.pkl')
tfidf_trans = joblib.load('tfidf_trans.pkl')
regression_pre = joblib.load('Regression_Predict_Rating.pkl') 


#read the data
housing_reviews = pd.read_table("reviews.csv", header='infer', delimiter=",", na_values='NaN')
listings = pd.read_table("listings_edited.csv", index_col=0, header='infer', delimiter=",")

#drop the missing value
drop_rating_index = housing_reviews.index[housing_reviews['rating'] == 0]
drop_rating_index.tolist()
housing_reviews = housing_reviews.drop(housing_reviews.index[drop_rating_index])
housing_reviews = housing_reviews[~housing_reviews['comments'].isnull()] 
housing_reviews = housing_reviews[~housing_reviews['rating'].isnull()] 
listings_edit = listings.drop(['state', 'city', 'zipcode', 'neighbourhood_cleansed', 'reviews_per_month'], axis=1)
listings_edit = listings_edit.dropna(subset=['host_response_time','host_response_rate','review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'])

#Fill in missing values
# rooms
listings_edit = listings_edit.fillna({"bathrooms": "0", "bedrooms": "0", "beds": "0"})
# fee
listings_edit = listings_edit.fillna({"price": "$0", "security_deposit": "$0", "cleaning_fee": "$0", "extra_people": "$0"})

#Transfer object to numeric values
obj_listings_edit = listings_edit.select_dtypes(include=['object']).copy()
#Transfer categorical to numerical values using manually input
# replace the value manually
response_time_num = {"host_response_time": {"within an hour": 0, "within a few hours": 1, 
                                                "within a day": 2, "a few days or more": 3, "none": 4}}
listings_edit.replace(response_time_num, inplace=True)
superhost_num = {"host_is_superhost": {"t": 1, "f": 0}}
listings_edit.replace(superhost_num, inplace=True)
#Transfer categorical to numerical values using sklearn.LabelEncoder
# host_id_verified
le = LabelEncoder()
listings_edit["host_identity_verified"] = le.fit_transform(listings_edit["host_identity_verified"])
# property_type_code
listings_edit["property_type"] = le.fit_transform(listings_edit["property_type"])
# room_type_code
listings_edit["room_type"] = le.fit_transform(listings_edit["room_type"])
# bed_type_code
listings_edit["bed_type"] = le.fit_transform(listings_edit["bed_type"])
#Transfer categorical to numerical values using pandas LabelEncoding
listings_edit["instant_bookable"] = listings_edit["instant_bookable"].astype('category')
listings_edit["instant_bookable"] = listings_edit["instant_bookable"].cat.codes
listings_edit["cancellation_policy"] = listings_edit["cancellation_policy"].astype('category')
listings_edit["cancellation_policy"] = listings_edit["cancellation_policy"].cat.codes

#Transfer strings to integers
listings_edit['bathrooms'] = listings_edit['bathrooms'].astype('int')
listings_edit['bedrooms'] = listings_edit['bedrooms'].astype('int')
listings_edit[ 'beds'] = listings_edit[ 'beds'].astype('int')
#Transfer percentages to integers
listings_edit['host_response_rate'] = listings_edit['host_response_rate'].str[:-1].astype('int')
#Transfer dollar prices to floats
listings_edit[['price']] = (listings_edit['price'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['security_deposit']] = (listings_edit['security_deposit'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['cleaning_fee']] = (listings_edit['cleaning_fee'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['extra_people']] = (listings_edit['extra_people'].replace( '[\$,)]','', regex=True ).astype(float))

#Extract amenities values and add new columns
# TV, wireless internet, air condition, heating, pets, washer, dryer
attrs = ['TV', 'Internet', 'Air conditioning', 'Kitchen' , 'Heating', 'Washer', 'Dryer']
rows = listings_edit.shape[0]
for attr in attrs:
    listings_edit[attr] = pd.Series(np.zeros(rows), index=listings_edit.index).astype(integer)

for index, row in listings_edit.iterrows():
    for attr in attrs:
        if (row['amenities'].find(attr)>=0):
            listings_edit.set_value(index, attr, 1)

listings_eval = listings_edit.drop(['listing_url','name','summary','picture_url','amenities', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'], axis=1)
#Item-Based Collaborative Filtering
listings_eval_update = listings_eval[['host_response_rate', 'host_is_superhost', 'host_identity_verified',
 'property_type', 'accommodates', 'bathrooms', 'beds', 'bed_type', 'price',
 'guests_included', 'minimum_nights', 'number_of_reviews', 'instant_bookable',
 'cancellation_policy', 'TV', 'Internet', 'Air conditioning', 'Kitchen',
 'Heating', 'Washer', 'Dryer']]

item_id_list = listings_eval_update.index.tolist()

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

#from StringIO import StringIO
#from PIL import Image
#import urllib

def recommend_item_list(query_index):
    euc_knn = NearestNeighbors(metric = 'euclidean', algorithm ='brute')
    euc_knn.fit(listings_eval_update)
    distances, indices = euc_knn.kneighbors(listings_eval_update.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print "Recommendation for {0}: \n".format(listings_eval_update.index[query_index])
        else:
            recommend_id = listings_eval_update.index[indices.flatten()[i]]
            print "{0}: 【{1}】\n    Name: '{2}'\n    Website: {3}\n\n    Summary: '{4}'\n".format(i, recommend_id,listings.loc[recommend_id]["name"],
             listings.loc[recommend_id]["listing_url"], listings.loc[recommend_id]["summary"]
                )
            #Image.open(StringIO(urllib.urlopen(listings.loc[recommend_id]["picture_url"]).read()))

def recommend_item():
    print ""
    print ""
    print '-------------------------------------'
    print "ITEM-BASED COLLABORATIVE RECOMMENDER"
    print '-------------------------------------'
    while True:
        print ""
        print '/////////////////////////////////////'
        print ""
        print "Please enter a listing ID, we will provide several top recommendations for you."
        print "Enter 'R' or 'r' to go back to the menu"
        print ""
        print "SOME LISTING IDs FOR TESTING:"
        print item_id_list[0:20]
        print ""

        intput = raw_input()
        i_index = 0
        if (intput == 'R' or intput == 'r'):
            break
        i_id = int(intput)
        if (i_id in item_id_list):
            i_index = item_id_list.index(i_id)
            recommend_item_list(i_index)
        else:
            print "Listing ID does not exist! Please check again."

def recommender_user(list_id, review_id, dataMat):
	print ""
	print ""
	print '-------------------------------------'
	print "USER-BASED COLLABORATIVE RECOMMENDER"
	print '-------------------------------------'
	while True:
		print ""
		print ""
		print "Please enter you Airbnb user name. If your user name is in our database, we will provide several top recommendations for you."
		print "Enter 'R' or 'r' to go back to the menu"
		print ""
		print "SOME REVIEW IDs FOR TESTING:"
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
			print "User name does not exist!"
			

words_to_remove=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


def clean_my_data(text):
    stemmer = PorterStemmer()
    clean_text=[]
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    container=[]
    for words in tagged:
        if (words[1][0] == 'N' or words[1][0]=='J' or words[1][0] == 'V') and (words[1][0] not in words_to_remove):
            container.append(words[0])
    for words in container:
        word = stemmer.stem(words)
        clean_text.append(word)
    return ' '.join(clean_text)



def predict_rating():
	print ""
	print ""
	print '--------------------------'
	print "SUGGESTED RATING FOR USER"
	print '--------------------------'
	while True:
		print ""
		print ""
		print "Please enter your review for Airbnb. The application will provide a suggested rating to you."
		print "Enter 'R' to go back to the menu"
		inp = raw_input()
		if (inp == 'R' or inp == 'r'):
			break
		print inp
		review = clean_my_data(inp)
		mat = CountV.transform([review])
		tfidf = tfidf_trans.transform(mat)
		rating = regression_pre.predict(tfidf)[0]
		print "SUGGESTED RATING:", rating


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

        if inp == '1':
            recommend_item()
    	if inp == '2':
    		recommender_user(list_id, review_id,dataMat)
    	if inp == '3':
    		predict_rating()
    	if (inp == 'Q' or inp == 'q'):
			break

recommenderinterface(list_id, review_id, dataMat)
