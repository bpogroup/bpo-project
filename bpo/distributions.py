import pandas
import random
import scipy
import numpy as np
from enum import Enum, auto
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor


class DistributionType(Enum):
    """An enumeration for different types of probability distribution."""
    CATEGORICAL = auto()
    """A categorical distribution.

    :meta hide-value:"""

    GAMMA = auto()
    """A gamma distribution.

    :meta hide-value:"""

    NORMAL = auto()
    """A normal distribution.

    :meta hide-value:"""

    BETA = auto()
    """A normal distribution.

    :meta hide-value:"""


class CategoricalDistribution:

    def __init__(self):
        self._values = []
        self._weights = []

    def learn(self, values, counts):
        self._values = values
        self._weights = counts

    def sample(self):
        return random.choices(self._values, weights=self._weights)[0]


class GammaDistribution:

    def __init__(self):
        self._alpha = 0
        self._loc = 0
        self._scale = 0

    def learn(self, values):
        fit_alpha, fit_loc, fit_scale = scipy.stats.gamma.fit(values)
        self._alpha = fit_alpha
        self._loc = fit_loc
        self._scale = fit_scale

    def sample(self):
        return scipy.stats.gamma.rvs(self._alpha, loc=self._loc, scale=self._scale)


class NormalDistribution:

    def __init__(self):
        self._mu = 0
        self._std = 0

    def learn(self, values):
        fit_mu, fit_std = scipy.stats.norm.fit(values)
        self._mu = fit_mu
        self._std = fit_std

    def sample(self):
        return scipy.stats.norm.rvs(self._mu, self._std)


class BetaDistribution:

    def __init__(self):
        self._a = 0
        self._b = 0
        self._loc = 0
        self._scale = 0

    def learn(self, values):
        fit_a, fit_b, fit_loc, fit_scale = scipy.stats.beta.fit(values)
        self._a = fit_a
        self._b = fit_b
        self._loc = fit_loc
        self._scale = fit_scale

    def sample(self):
        return scipy.stats.beta.rvs(self._a, self._b, self._loc, self._scale)


class StratifiedNumericDistribution:

    def __init__(self):
        self._target_column = ''
        self._feature_columns = []
        self._onehot_columns = []
        self._standardization_columns = []
        self._rest_columns = []

        self._normalizer = None
        self._standardizer = None
        self._encoder = None
        self._regressor = None

        self._stratifier = ''
        self._stratified_errors = dict()
        self._overall_mean = 0

    # onehot_columns will be onehot encoded, standardization_columns will be Z-Score normalized
    # all other features will be minmax normalized.
    def learn(self, data, target_column, feature_columns, onehot_columns, standardization_columns, stratifier):
        x = data[feature_columns]
        y = data[target_column]

        self._normalizer = MinMaxScaler()
        self._standardizer = StandardScaler()
        self._encoder = OneHotEncoder(sparse=False)

        self._target_column = target_column
        self._feature_columns = feature_columns
        self._onehot_columns = onehot_columns
        self._standardization_columns = standardization_columns
        self._rest_columns = [col for col in feature_columns if col not in standardization_columns and col not in onehot_columns]

        self._overall_mean = y.mean()

        standardized_data = self._standardizer.fit_transform(x[self._standardization_columns])
        normalized_data = self._normalizer.fit_transform(x[self._rest_columns])
        onehot_data = self._encoder.fit_transform(x[self._onehot_columns])

        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)

        self._regressor = MLPRegressor(hidden_layer_sizes=(x.shape[1], int(x.shape[1]/2), int(x.shape[1]/4)), activation='relu', solver='adam').fit(x, y)

        # now calculate the errors
        self._stratifier = stratifier
        df_error = data[[self._stratifier]].copy()
        df_error['y'] = data[target_column]
        df_error['y_hat'] = list(self._regressor.predict(x))
        df_error['error'] = df_error['y'] - df_error['y_hat']

        overall_value = NormalDistribution()
        overall_value.learn(list(df_error['error']))

        possible_values = data[stratifier].unique()
        for pv in possible_values:
            self._stratified_errors[pv] = NormalDistribution()
            stratified_errors = list(df_error[df_error[self._stratifier] == pv]['error'])
            if len(stratified_errors) > 50:
                self._stratified_errors[pv].learn(stratified_errors)
            else:
                self._stratified_errors[pv] = overall_value

    # features is a dictionary that maps feature labels to lists of values
    def sample(self, features):
        data = pandas.DataFrame(features, index=[1])

        standardized_data = self._standardizer.transform(data[self._standardization_columns])
        normalized_data = self._normalizer.transform(data[self._rest_columns])
        onehot_data = self._encoder.transform(data[self._onehot_columns])

        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)

        processing_time = self._regressor.predict(x)[0]
        if processing_time <= 0:
            processing_time = self._overall_mean
        error = self._stratified_errors[features[self._stratifier]].sample()
        max_retries = 10
        retry = 0
        while retry < max_retries and processing_time + error <= 0:
            error = self._stratified_errors[features[self._stratifier]].sample()
            retry += 1
        if processing_time + error > 0:
            return processing_time + error
        else:
            return processing_time
