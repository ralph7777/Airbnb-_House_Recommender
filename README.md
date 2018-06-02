
# Airbnb_House_Recommender

Feel free to add any ideas or works you have done along with name and date.

-------
2018/6/2 Lanny

Implement and finish the item-based recommender for Python application，also optimized the outputs for better looking<br />
Move around files and rename folders for easy access. <br />

-------
2018/6/1 Ralph

Modify the function provided by professor, extract the computation of column-based similarity. <br />
Re-run and clean the code based on selected features. <br />
 
-------
2018/5/31 Ralph

Add Percentile-MAE figures to visualize the data. <br />
Add MAE results for testing dataset using different FS approach. <br />
Compare results and decide to choose 15 critical features. <br />
Clean codes. <br />

-------
2018/5/30 Lanny

Because have not decide how to evaluate accuracy fot the feature selection, just chose to use one of the selcted features to start Item-Based filtering for now: <br />
Use 3 types of item similarity metrics:<br />
 #1 Cosine Similarity <br />
 #2 Pearson Similarity <br />
 #3 Euclidean Similarity <br />

Calculate MAE and pick Euclidean for this project.<br />

Next:<br />
find a proper way to evaluate the accuracy of selected features.<br />

-------
2018/5/29 Lanny

Did 80-20 splitting. Evaluate some of the accuracy.<br />

Add 2 more basic feature methods to do feature selection.<br />

Next:<br />
find a proper way to evaluate the accuracy.<br />

For Item-Based filtering: use 3 types of item similarity metrics:<br />

 #1 Cosine Similarity <br />
 #2 Adjusted Cosine Similarity <br />
 #3 Pearson Similarity <br />

-------
2018/5/25 Ralph

Split the data attributes from the target (review_scores_rating). <br />

Add three basic feature methods to do feature selection.<br />

Reduce the number of features from 28 to 18 or 17.<br />

Next:<br />

Add more systemetic regression analysis (more model and parameter analysis), to determine the best model for rating prediction.<br />

Do 80-20 splitting analysis so that the accuray of predicting model can be evaluated.<br />

-------
2018/5/22 Ralph

Finish attributes preprocessing and generate the table for feature selection.<br />

When reading in data, set id as index column<br />

Transfer 'cancellation_policy"<br />

Extract ['TV', 'Internet', 'Air conditioning', 'Kitchen' , 'Heating', 'Washer', 'Dryer'] in amenities to be new columns with values of 0 and 1<br />

Reorder some codes<br />


-------
2018/5/21 Lanny

Almost done with attributes preprocessing. 
Except for Amenities.

Used 3 methods to transfer categorical to numerical:
host_response_time<br />
host_is_superhost, host_identity_verified<br />
property_type, room_type, bed_type <br />
instant_bookable <br />
Did not do "cancellation_policy" <br />

Dropped 'listing_url', 'name', 'summary', 'picture_url', 'state', 'city', 'zipcode', 'neighbourhood_cleansed', 'reviews_per_month' <br />

Dropped NaN for review_scores_rating/ review_scores_accuracy/_cleanliness/_checkin/_communication/_location/_value <br />


Transferred 'host_response_rate' to integers<br />
Transferred 'price', 'security_deposit', 'cleaning_fee', 'extra_people' to float<br />
bathrooms, bedrooms, beds: fill missing value with 0. 

-------
2018/5/20 Ralph

Attributes preprocessing:

id - 1<br />
Listing_url, Name, Summary, Picture_url - 5<br />
host_response_time - 2<br />
host_response_rate - 3<br />
host_is_superhost, host_identity_verified - 2<br />
neighbourhood_cleansed, state, city, zipcode - 6, 7<br />
property_type, room_type - 2<br />
accommodates, bathrooms, bedrooms, beds - 1, 7<br />
amenities (TV,  wireless internet, air condition, heating, pets, washer, dryer) - 4<br />
price, security_deposit, cleaning_fee, extra_people - 3<br />
minimum_nights, maximum_nights, number_of_reviews - 1<br />
review_scores_rating - 7<br />
review_scores_accuracy/_cleanliness/_checkin/_communication/_location/_value - 6<br />
instant_bookable - 2<br />
cancellation_policy - 2<br />
reviews_per_month - 6<br />

1. Ready for use.
2. Transfer categorical to numerical.
3. Transfer to integers.
4. Need text resolution.
5. Extract for late use.
6. Not include.
7. Handle missing values. *

For handling missing values：
1. state, city: shall we remove the houses not in IL 3/5207 and Chicago 31/5207? 
2. bathrooms, bedrooms, beds: fill missing value with 0.
3. review_scores_rating： remove those houses without rating scores


-------
2018/5/20 Ralph

Data Preprocessing:

Current attributes statistic as follow

![Attributes](/Attributes.jpg?raw=true "Attributes")

From top to bottom:
1. Id, Listing_url, Name, Summary, Picture_url: we don't need these for analysis, extract to different table
2. host_response_time: categorical to numerical
3. host_response_rate: transfer to numerical
4. host_is_superhost, host_identity_verified: binary, categorical to numerical 0 and 1?
5. neighbourhood_cleansed, zipcode: similar, pick one?
6. state, city: other states? shall we delete those houses?
7. property_type, room_type: categorical to numerical
8. accommodates, bathrooms, bedrooms, beds: numerical, for missing values shall we use 0?
9. amenities: text tokenize to multiple attributes - TV,  wireless internet, air condition, heating, pets, washer, dryer
10. price, security_deposit, cleaning_fee, extra_people: transfer to numerical
11. minimum_nights, maximum_nights, number_of_reviews: just records in system, not important for prediction?
12. review_scores_rating: 4469 out of 5207, we can set those with 0 review for prediction (without verification)
13. review_scores_accuracy/_cleanliness/_checkin/_communication/_location/_value: should not be used for prediction?
14. reviews_per_month: not important for predictation?

Looks like lots of preprocessing work to be done... manually.

-------
2018/5/19 Ralph

## Design Features:
### 1. Recommender:
Item-based recommender:<br />
a) Do feature selection based on rating value, determine the relative important features for rating. <br />
b) Compare the similaries among selected features of items in system, give the most similar item to the user choice as recommender.<br />

User-based recommender:<br />
a) Transfer user's text review to rating score based on key words.<br />
b) Look for users with similar ratings on same items, recommend item to current user based on other's choice.<br />

### 2. Review Text Mining (Interface feature):
Make a prediction of item score based on content of text review.

### 3. Rent Price Advicing (Interface feature):
Based on prices of similar item (determined by selected features similarity), give reasonable price for a new item.
Here we may need to switch target attribute to Price.
