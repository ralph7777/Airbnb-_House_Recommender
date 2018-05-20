2018/5/20 Ralph

## Design Features:
### 1. Recommender:
Item-based recommender:
a) Do feature selection based on rating value, determine the relative important features for rating. 
b) Compare the similaries among selected features of items in system, give the most similar item to the user choice as recommender.

User-based recommender:
a) Transfer user's text review to rating score based on key words.
b) Look for users with similar ratings on same items, recommend item to current user based on other's choice.

### 2. Review Text Mining (Interface feature):
Make a prediction of item score based on content of text review.

### 3. Rent Price Advicing (Interface feature):
Based on prices of similar item (determined by selected features similarity), give reasonable price for a new item.
Here we may need to switch target attribute to Price.

-------
2018/5/20 Ralph

Data Preprocessing:

Current attributes statistic as follow

![Attributes](/Attributes.jpg?raw=true "Attributes")

From top to bottom:
1. Id, Listing_url,Name, Summary, Picture_url: we don't need for processing.
2. host_response_time: categorical to numerical
3. host_response_rate: transfer to numerical
4. host_is_superhost, host_identity_verified: binary, transfer to 0 and 1
5. neighbourhood_cleansed, zipcode: similar,  pick one
6. state, city: other states? shall we delete those items?
7. property_type,room_type: categorical to numerical
8. accommodates, bathrooms, bedrooms, beds: numerical, for missing values shall we use 0?
9. amenities: text tokenize to multiple attributes
10. price, security_deposit, cleaning_fee, extra_people: transfer to numerical
11. minimum_nights, maximum_nights, number_of_reviews: just records in system, not important for predictation?
12. review_scores_rating: 4469 out of 5207, we can set those with 0 review for prediction (without verification)
13. review_scores_accuracy/_cleanliness/_checkin/_communication/_location/_value: should not be used for prediction?
14. reviews_per_month: not important for predictation?

Lots of preprocessing work to be done... manually.


