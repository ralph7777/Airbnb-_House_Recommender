# coding: utf-8

# Airbnb House Analysis and Recommender Application

##### Jie Bao, Kuangyi Zhang, Lanny Xu
##### Dr. Bamshad Mobasher, Spring 2018

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

from PIL import Image
import urllib, cStringIO

from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


# read in tables
housing_reviews = pd.read_table("reviews.csv", header='infer', delimiter=",", na_values='NaN')
listings = pd.read_table("listings_edited.csv", index_col=0, header='infer', delimiter=",")

# read in established models
CountV = joblib.load('CountVectorizer.pkl')
tfidf_trans = joblib.load('tfidf_trans.pkl')
regression_pre = joblib.load('Regression_Predict_Rating.pkl') 

# drop missing value
drop_rating_index = housing_reviews.index[housing_reviews['rating'] == 0]
drop_rating_index.tolist()
housing_reviews = housing_reviews.drop(housing_reviews.index[drop_rating_index])
housing_reviews = housing_reviews[~housing_reviews['comments'].isnull()] 
housing_reviews = housing_reviews[~housing_reviews['rating'].isnull()] 
listings_edit = listings.drop(['state', 'city', 'zipcode', 'neighbourhood_cleansed', 'reviews_per_month'], axis=1)
listings_edit = listings_edit.dropna(subset=['host_response_time','host_response_rate','review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'])

# fill in missing values
# rooms
listings_edit = listings_edit.fillna({"bathrooms": "0", "bedrooms": "0", "beds": "0"})
# fee
listings_edit = listings_edit.fillna({"price": "$0", "security_deposit": "$0", "cleaning_fee": "$0", "extra_people": "$0"})

# transfer object to numeric values
obj_listings_edit = listings_edit.select_dtypes(include=['object']).copy()
# transfer categorical to numerical values using manually input
response_time_num = {"host_response_time": {"within an hour": 0, "within a few hours": 1, 
                                                "within a day": 2, "a few days or more": 3, "none": 4}}
listings_edit.replace(response_time_num, inplace=True)
superhost_num = {"host_is_superhost": {"t": 1, "f": 0}}
listings_edit.replace(superhost_num, inplace=True)

# transfer categorical to numerical values using sklearn.LabelEncoder
# host_id_verified
le = LabelEncoder()
listings_edit["host_identity_verified"] = le.fit_transform(listings_edit["host_identity_verified"])
# property_type_code
listings_edit["property_type"] = le.fit_transform(listings_edit["property_type"])
# room_type_code
listings_edit["room_type"] = le.fit_transform(listings_edit["room_type"])
# bed_type_code
listings_edit["bed_type"] = le.fit_transform(listings_edit["bed_type"])

# transfer categorical to numerical values using pandas LabelEncoding
listings_edit["instant_bookable"] = listings_edit["instant_bookable"].astype('category')
listings_edit["instant_bookable"] = listings_edit["instant_bookable"].cat.codes
listings_edit["cancellation_policy"] = listings_edit["cancellation_policy"].astype('category')
listings_edit["cancellation_policy"] = listings_edit["cancellation_policy"].cat.codes

# transfer strings to integers
listings_edit['bathrooms'] = listings_edit['bathrooms'].astype('int')
listings_edit['bedrooms'] = listings_edit['bedrooms'].astype('int')
listings_edit[ 'beds'] = listings_edit[ 'beds'].astype('int')

# transfer percentages to integers
listings_edit['host_response_rate'] = listings_edit['host_response_rate'].str[:-1].astype('int')

# transfer dollar prices to floats
listings_edit[['price']] = (listings_edit['price'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['security_deposit']] = (listings_edit['security_deposit'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['cleaning_fee']] = (listings_edit['cleaning_fee'].replace( '[\$,)]','', regex=True ).astype(float))
listings_edit[['extra_people']] = (listings_edit['extra_people'].replace( '[\$,)]','', regex=True ).astype(float))

# extract amenities values and add new columns: TV, wireless internet, air condition, heating, pets, washer, dryer
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

# select the user with 3 or more comments for the house listing
recommend = housing_reviews[['listing_id', 'reviewer_id', 'rating']]
upper_2 = recommend['reviewer_id'].value_counts() > 3
users = upper_2[upper_2].index.tolist()
recommend = recommend[recommend['reviewer_id'].isin(users)]

# extract user_id and listing_id 
list_id = recommend.iloc[0:,0].unique()
review_id = recommend.iloc[0:,1].unique()

# processing the data to build the matrix
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



# three distance meansurement
def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

# function to compute similaries among users
import operator
def most_similar_users(dataMat, queryUser, K, metric = pearsSim):
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
    return sorted_keys

# function to compute rating based on similar users
def predict(dataMat, review_id, list_index, similar_users, K):
    if len(similar_users) == 0:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    index = 0
    for key, values in similar_users:
        if(index == K):
            break
        if dataMat[key][list_index] != 0:
            neighbor_id = review_id[key]        
            neighbor_similarity = values
            rating = dataMat[key][list_index]
            numerator += neighbor_similarity * rating
            denominator += neighbor_similarity
            index += 1
    result = numerator/denominator
    return result 


# function to recommend top K listings (items) based on user similarity
def recommend_list(dataMat, user_id, list_id, queryUser, K):
    nopreference_list = np.where(dataMat[queryUser,:]==0)[0]
    predict_rating = {}
    sorted_most_similar_users = most_similar_users(dataMat, queryUser, K, metric=pearsSim)
    for item in nopreference_list:
        result = predict(dataMat, user_id, item, sorted_most_similar_users, K)
        predict_rating[item] = result
    sorted_list = sorted(predict_rating.items(), key=operator.itemgetter(1), reverse=True)
    print ""
    print '-----------------------'
    print 'Selected User ID:', user_id[queryUser]
    print '-----------------------'
    print 'Listing recommendations:\n'

    index = 1
    for key, values in sorted_list:
        if(index > K):
            break
        list_index = list_id[key]
        print "Recommendation %s:\nid: %s\nName: '%s'\nWebsite: %s\nSummary: '%s'\n" %(index, list_index, listings.loc[list_index]["name"], listings.loc[list_index]["listing_url"], listings.loc[list_index]["summary"])
        imgFile = cStringIO.StringIO(urllib.urlopen(listings.loc[list_index]["picture_url"]).read())
        img = Image.open(imgFile)
        img.show()
        print '------------------------------------------'
        index += 1

# function to recommend top 5 listings based on listing similarity
def recommend_item_list(query_index):
    euc_knn = NearestNeighbors(metric = 'euclidean', algorithm ='brute')
    euc_knn.fit(listings_eval_update)
    distances, indices = euc_knn.kneighbors(listings_eval_update.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print ""
            print '--------------------------'
            print 'Selected Listing ID:', listings_eval_update.index[query_index]
            print '--------------------------'
            print 'Listing recommendations:\n'
        else:
            recommend_id = listings_eval_update.index[indices.flatten()[i]]
            print "Recommendation %s:\nid: %s\nName: '%s'\nWebsite: %s\nSummary: '%s'\n" %(i, recommend_id,listings.loc[recommend_id]["name"], listings.loc[recommend_id]["listing_url"], listings.loc[recommend_id]["summary"])
            imgFile = cStringIO.StringIO(urllib.urlopen(listings.loc[recommend_id]["picture_url"]).read())
            img = Image.open(imgFile)
            img.show()
            print '------------------------------------------'

# Recommender UI for user prompt            
def recommend_item():
    print ""
    print ""
    print '======================'
    print "Item-based Recommender"
    print '======================'
    while True:
        print ""
        print "Please enter a Listing ID. System will provide top recommendations for you."
        print "Enter 'R' or 'r' to go back to main menu."
        print ""
        print "* some Listing IDs for testing:"
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
            print "Listing ID does not exist! Please check it again."

def recommender_user(list_id, review_id, dataMat):
	print ""
	print ""
	print '==================================='
	print "User-based Collabrative Recommender"
	print '==================================='
	while True:
		print ""
		print "Please enter your Airbnb User ID. System will provide top recommendations for you."
		print "Enter 'R' or 'r' to go back to main menu."
		print ""
		print "* some User IDs for testing:"
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
			print "User ID does not exist! Please check it again."
        print ""
			

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

# Text Mining UI for user prompt
def predict_rating():
	print ""
	print ""
	print '======================='
	print "User Review Text Mining"
	print '======================='
	while True:
		print ""
		print "Please enter your review for Airbnb listing. System will suggest a rating score (1-5) to you."
		print "Enter 'R' to go back to the menu"
		inp = raw_input()
		if (inp == 'R' or inp == 'r'):
			break
		print inp
		review = clean_my_data(inp)
		mat = CountV.transform([review])
		tfidf = tfidf_trans.transform(mat)
		rating = regression_pre.predict(tfidf)[0]
		print "Rating suggestion:", rating

# Main Menu UI
def recommenderinterface(list_id, review_id, dataMat):
    while True:
    	print ""
    	print ""
    	print "===================================================================="
    	print "***** Airbnb House Recommender/Rating Predication Application ******"
    	print "===================================================================="
    	print ""
    	print "This application consists of two Recommendation Systems and one Rating Prediction System:\n"
    	print "1. Item-based Recommender"
    	print "2. User-based Collaborative Recommender"
    	print "3. User Review Rating Prediction (T)"
    	print ""
    	print "Enter '1' or '2' or '3' for different functionality"
    	print "Enter 'Q' or 'q' to quit application\n"
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
