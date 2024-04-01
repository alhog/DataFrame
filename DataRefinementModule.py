import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class DataRefinementModule:
    def __init__(self, config):
        self.config = config

    def normalize_data(self, data, method='minmax', cols=None):
        """Normalizes numerical data using the specified method."""
        if cols is None:
            cols = data.select_dtypes(include=['float64', 'int64']).columns

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported method: {method}")

        data[cols] = scaler.fit_transform(data[cols])
        return data

    def reduce_dimensionality(self, data, method='pca', n_components=None):
        """Reduces the dimensionality of the data using the specified method."""
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unsupported method: {method}")

        reduced_data = pd.DataFrame(reducer.fit_transform(data), index=data.index)
        return reduced_data

    def select_features(self, data, target, method='univariate', n_features=None):
        """Selects the most relevant features from the dataset."""
        if method == 'univariate':
            if data[target].dtype.kind == 'O':
                scorer = f_classif
            else:
                scorer = f_regression
            selector = SelectKBest(scorer, k=n_features)
        elif method == 'recursive':
            estimator = LogisticRegression(random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
        else:
            raise ValueError(f"Unsupported method: {method}")

        selected_features = selector.fit_transform(data.drop(target, axis=1), data[target])
        selected_cols = data.drop(target, axis=1).columns[selector.get_support()]
        return pd.DataFrame(selected_features, columns=selected_cols, index=data.index)

    def sample_data(self, data, target, method='random', sampling_strategy=None):
        """Samples or subsets the data for efficient processing or balanced class distributions."""
        if method == 'random':
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        elif method == 'stratified':
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_idx, test_idx in split.split(data, data[target]):
                train_data = data.loc[train_idx]
                test_data = data.loc[test_idx]
        elif method == 'oversampling':
            oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X, y = data.drop(target, axis=1), data[target]
            X_resampled, y_resampled = oversampler.fit_resample(X, y)
            train_data = pd.concat([X_resampled, y_resampled], axis=1)
        elif method == 'undersampling':
            undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X, y = data.drop(target, axis=1), data[target]
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            train_data = pd.concat([X_resampled, y_resampled], axis=1)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if method in ['random', 'stratified']:
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            return train_data, test_data
        else:
            return train_data

    def transform_data(self, data, transformations):
        """Applies additional data transformations or encodings."""
        for transform in transformations:
            if transform['type'] == 'log':
                data[transform['columns']] = np.log1p(data[transform['columns']])
            elif transform['type'] == 'boxcox':
                from scipy.stats import boxcox
                data[transform['columns']] = boxcox(data[transform['columns']] + 1, lmbda=transform['lambda'])
            elif transform['type'] == 'binarize':
                data[transform['columns']] = (data[transform['columns']] > transform['threshold']).astype(int)
            # Add more transformation types as needed
        return data

    def refine_data(self, data, target):
        """Orchestrates the data refinement process."""
        # Normalize numerical data
        data = self.normalize_data(data, method='minmax')

        # Reduce dimensionality
        data = self.reduce_dimensionality(data, method='pca', n_components=0.8)

        # Select relevant features
        data = self.select_features(data, target, method='univariate', n_features=20)

        # Sample data for balanced class distribution
        train_data, test_data = self.sample_data(data, target, method='stratified')

        # Apply additional transformations
        train_data = self.transform_data(train_data, [
            {'type': 'log', 'columns': ['income']},
            {'type': 'binarize', 'columns': ['is_employed'], 'threshold': 0.5}
        ])

        return train_data, test_data
