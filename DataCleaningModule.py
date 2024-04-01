import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

class DataCleaningModule:
    def __init__(self, config):
        self.config = config

    def handle_missing_data(self, data, strategy='mean', cols=None):
        """Handles missing data using the specified strategy."""
        if cols is None:
            cols = data.columns

        if strategy == 'mean':
            return data[cols].fillna(data[cols].mean())
        elif strategy == 'median':
            return data[cols].fillna(data[cols].median())
        elif strategy == 'mode':
            return data[cols].fillna(data[cols].mode().iloc[0])
        elif strategy == 'interpolate':
            return data[cols].interpolate()
        elif strategy == 'drop':
            return data.dropna(subset=cols)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def detect_outliers(self, data, method='zscore', threshold=3):
        """Detects outliers using the specified method and threshold."""
        if method == 'zscore':
            outliers = (np.abs(zscore(data)) > threshold).any(axis=1)
        elif method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return outliers

    def remove_outliers(self, data, outliers):
        """Removes outliers from the dataset."""
        return data.loc[~outliers]

    def cap_outliers(self, data, outliers, method='iqr'):
        """Caps outliers using the specified method."""
        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            data.loc[data < (q1 - 1.5 * iqr)] = q1 - 1.5 * iqr
            data.loc[data > (q3 + 1.5 * iqr)] = q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unsupported method: {method}")
        return data

    def remove_duplicates(self, data, subset=None, keep='first'):
        """Removes duplicate rows or records from the dataset."""
        return data.drop_duplicates(subset=subset, keep=keep)

    def convert_data_types(self, data, type_mapping):
        """Converts data types of columns based on the provided mapping."""
        for col, dtype in type_mapping.items():
            data[col] = data[col].astype(dtype)
        return data

    def format_data(self, data, format_mapping):
        """Formats data columns based on the provided mapping."""
        for col, format_func in format_mapping.items():
            data[col] = data[col].apply(format_func)
        return data

    def normalize_data(self, data, method='minmax', cols=None):
        """Normalizes numerical data using the specified method."""
        if cols is None:
            cols = data.select_dtypes(include=['float64', 'int64']).columns

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported method: {method}")

        data[cols] = scaler.fit_transform(data[cols])
        return data

    def validate_data(self, data, constraints):
        """Validates the data against predefined constraints."""
        for col, constraint in constraints.items():
            if 'dtype' in constraint:
                if not data[col].dtypes == constraint['dtype']:
                    raise ValueError(f"Data type mismatch for column '{col}': expected {constraint['dtype']}, found {data[col].dtypes}")

            if 'min' in constraint:
                if data[col].min() < constraint['min']:
                    raise ValueError(f"Minimum value violation for column '{col}': expected minimum {constraint['min']}, found {data[col].min()}")

            if 'max' in constraint:
                if data[col].max() > constraint['max']:
                    raise ValueError(f"Maximum value violation for column '{col}': expected maximum {constraint['max']}, found {data[col].max()}")

            if 'nulls' in constraint:
                if constraint['nulls'] == 'not_allowed' and data[col].isnull().any():
                    raise ValueError(f"Null values found in column '{col}', but not allowed")

            # Add more constraint types as needed
        return data

    def clean_data(self, data):
        """Orchestrates the data cleaning process."""
        # Handle missing data
        data = self.handle_missing_data(data, strategy='mean')

        # Detect and handle outliers
        outliers = self.detect_outliers(data, method='zscore')
        data = self.remove_outliers(data, outliers)

        # Remove duplicates
        data = self.remove_duplicates(data, subset=['id'])

        # Convert data types
        data = self.convert_data_types(data, {'age': 'int64', 'income': 'float64'})

        # Format data
        data = self.format_data(data, {'date': lambda x: pd.to_datetime(x)})

        # Normalize numerical data
        data = self.normalize_data(data, method='minmax', cols=['age', 'income'])

        # Validate data
        constraints = {
            'age': {'dtype': 'int64', 'min': 0, 'max': 120, 'nulls': 'not_allowed'},
            'income': {'dtype': 'float64', 'min': 0, 'nulls': 'allowed'},
            'date': {'dtype': 'datetime64[ns]', 'nulls': 'not_allowed'}
        }
        data = self.validate_data(data, constraints)

        return data
