import pandas as pd
import h3

airbnb_calendar_dataset = pd.read_csv('data/airbnb/base_data/calendar.csv')
airbnb_listing_dataset = pd.read_csv('data/airbnb/base_data/listings.csv')
airbnb_listing_dataset = airbnb_listing_dataset[['id', 'host_is_superhost', 'host_identity_verified', 'neighbourhood_cleansed', 'zipcode', 'longitude', 'latitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included', 'number_of_reviews', 'review_scores_rating', 'cancellation_policy', 'reviews_per_month']]
airbnb_dataset = airbnb_calendar_dataset.merge(right=airbnb_listing_dataset, how='left', left_on='listing_id', right_on='id')
airbnb_dataset = airbnb_dataset.dropna()

airbnb_dataset['ID'] = airbnb_dataset.apply(lambda row: h3.geo_to_h3(lat=row['latitude'], lng=row['longitude'], resolution=9), axis=1)
airbnb_dataset['price'] = airbnb_dataset['price'].str.replace('$', '', regex=False)
airbnb_dataset['price'] = airbnb_dataset['price'].str.replace(',', '')
airbnb_dataset = airbnb_dataset.astype({'listing_id': object, 'price': float})
airbnb_dataset['date'] = pd.to_datetime(airbnb_dataset['date'])
airbnb_dataset['date'] = airbnb_dataset['date'].dt.strftime("%Y-%m-01")
airbnb_dataset = airbnb_dataset.drop(columns=['id', 'available', 'longitude', 'latitude', 'listing_id'])
airbnb_dataset.to_parquet("data/airbnb/ml_data/airbnb_base.parquet")