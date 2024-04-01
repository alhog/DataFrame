import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import geopy
from geopy.geocoders import Nominatim
from datetime import datetime

class DataEnrichmentModule:
    def __init__(self, config):
        self.config = config
        self.geolocator = Nominatim(user_agent="data_enrichment")

    def engineer_features(self, data, operations):
        """Creates new features by applying mathematical or statistical operations."""
        for operation in operations:
            if operation['type'] == 'polynomial':
                poly = PolynomialFeatures(degree=operation['degree'])
                data[operation['output_cols']] = poly.fit_transform(data[operation['input_cols']])
            elif operation['type'] == 'interaction':
                data[operation['output_col']] = data[operation['input_cols']].product(axis=1)
            # Add more feature engineering techniques as needed
        return data

    def encode_categorical_data(self, data, encoding_mapping):
        """Encodes categorical variables into numerical representations."""
        for col, encoding in encoding_mapping.items():
            if encoding == 'one_hot':
                ohe = OneHotEncoder(handle_unknown='ignore')
                data = pd.concat([data, pd.DataFrame(ohe.fit_transform(data[[col]]).toarray(),
                                                     columns=[f"{col}_{val}" for val in ohe.categories_[0]])], axis=1)
                data = data.drop(col, axis=1)
            elif encoding == 'label':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            # Add more encoding techniques as needed
        return data

    def preprocess_text(self, data, text_col, tokenize=True, stem=False, lemmatize=False, remove_stopwords=False, ngrams=(1, 1)):
        """Preprocesses and transforms unstructured text data."""
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def preprocess(text):
            tokens = word_tokenize(text.lower())
            if remove_stopwords:
                tokens = [token for token in tokens if token not in stop_words]
            if stem:
                tokens = [stemmer.stem(token) for token in tokens]
            if lemmatize:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            if tokenize:
                return tokens
            else:
                return ' '.join(tokens)

        if ngrams[0] == ngrams[1] == 1:
            data[text_col] = data[text_col].apply(preprocess)
        else:
            data[f"{text_col}_ngrams"] = data[text_col].apply(lambda text: [' '.join(ngram) for ngram in zip(*(preprocess(text)[i:] for i in range(ngrams[0])))])

        return data

    def integrate_external_data(self, data, source, join_cols):
        """Integrates the dataset with external data sources."""
        # Implement data integration logic (e.g., SQL queries, API calls)
        pass

    def enrich_geospatial_data(self, data, address_col):
        """Enriches the dataset with geospatial information."""
        data[['latitude', 'longitude']] = data[address_col].apply(self.geocode_address)
        data['location'] = list(zip(data['latitude'], data['longitude']))
        data['distance_from_origin'] = data['location'].apply(self.calculate_distance_from_origin)
        return data

    def geocode_address(self, address):
        """Geocodes an address and returns its latitude and longitude."""
        location = self.geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return np.nan, np.nan

    def calculate_distance_from_origin(self, location):
        """Calculates the distance from a given location to the origin."""
        origin = (0, 0)
        return geopy.distance.geodesic(origin, location).km

    def enrich_time_series_data(self, data, date_col):
        """Enriches time-series data with temporal features."""
        data[date_col] = pd.to_datetime(data[date_col])
        data['year'] = data[date_col].dt.year
        data['month'] = data[date_col].dt.month
        data['day'] = data[date_col].dt.day
        data['dayofweek'] = data[date_col].dt.dayofweek
        data['quarter'] = data[date_col].dt.quarter
        # Add more time-based transformations as needed
        return data

    def enrich_data(self, data):
        """Orchestrates the data enrichment process."""
        # Feature engineering
        data = self.engineer_features(data, [
            {'type': 'polynomial', 'input_cols': ['age', 'income'], 'output_cols': ['poly_features'], 'degree': 2},
            {'type': 'interaction', 'input_cols': ['age', 'income'], 'output_col': 'age_income_interaction'}
        ])

        # Categorical data encoding
        data = self.encode_categorical_data(data, {'gender': 'one_hot', 'education_level': 'label'})

        # Text preprocessing
        data = self.preprocess_text(data, 'description', tokenize=True, stem=True, lemmatize=True, remove_stopwords=True, ngrams=(1, 2))

        # Integrate external data
        # data = self.integrate_external_data(data, 'external_source', ['id'])

        # Geospatial data enrichment
        data = self.enrich_geospatial_data(data, 'address')

        # Time series data enrichment
        data = self.enrich_time_series_data(data, 'date')

        return data
