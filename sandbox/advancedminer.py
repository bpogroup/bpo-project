import random
import pandas
import datetime
from statistics import mean
from enum import Enum, auto
import scipy
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
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
        return random.sample(self._values, self._weights)


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
        data = pandas.DataFrame(features)

        standardized_data = self._standardizer.transform(data[self._standardization_columns])
        normalized_data = self._normalizer.transform(data[self._rest_columns])
        onehot_data = self._encoder.transform(data[self._onehot_columns])

        x = np.concatenate([standardized_data, normalized_data, onehot_data], axis=1)

        processing_time = self._regressor.predict(x)[0]
        if processing_time <= 0:
            print("Using overall mean")
            processing_time = self._overall_mean
        error = self._stratified_errors[features[self._stratifier][0]].sample()
        if processing_time + error > 0:
            return processing_time + error
        else:
            print("Infeasible error")
            return processing_time

# Only cases that start on or after earliest_start and complete on or after latest_completion will be taken into account.
# earliest_start and latest completion are datetime objects. They should either both be None or both have a value.
# datafields are a mapping field name to DistributionType, where field name is one of the columns of the log.
# for the corresponding log column, a data type will be learned according to the specified distribution.

def mine_problem(log, task_type_filter=None, datetime_format="%Y/%m/%d %H:%M:%S", earliest_start=None, latest_completion=None, min_resource_count=2, resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168, datafields=dict()):
    df = log.copy()
    df['Start Timestamp'] = pandas.to_datetime(df['Start Timestamp'], format=datetime_format)
    df['Complete Timestamp'] = pandas.to_datetime(df['Complete Timestamp'], format=datetime_format)
    df_cases = df.groupby('Case ID').agg(case_start=('Start Timestamp', 'min'), case_complete=('Start Timestamp', 'min'), trace=('Activity', lambda tss: list(tss)))
    if earliest_start is not None and latest_completion is not None:
        df_cases = df_cases[(df_cases['case_start'] >= earliest_start) & (df_cases['case_complete'] <= latest_completion)]
        relevant_ids = list(df_cases.index)
        df = df[df['Case ID'].isin(relevant_ids)]

    # Filter the data to the relevant columns and add the durations to the tasks
    df = df[['Case ID', 'Activity', 'Resource', 'Start Timestamp', 'Complete Timestamp'] + list(datafields.keys())]
    df['Duration'] = df[['Start Timestamp', 'Complete Timestamp']].apply(lambda tss: (tss[1]-tss[0]).total_seconds()/3600, axis=1)

    # Sort the data
    df = df.sort_values(by=['Case ID', 'Complete Timestamp'])

    # Mine the datatypes and corresponding distributions
    data_types = dict()
    for datafield in datafields:
        if datafields[datafield] == DistributionType.CATEGORICAL:
            distribution = CategoricalDistribution()
            distribution.learn(list(df[datafield].value_counts().index), list(df[datafield].value_counts().values))
            data_types[datafield] = distribution
        elif datafields[datafield] == DistributionType.GAMMA:
            distribution = GammaDistribution()
            distribution.learn(list(df[datafield]))
            data_types[datafield] = distribution
        elif datafields[datafield] == DistributionType.NORMAL:
            distribution = NormalDistribution()
            distribution.learn(list(df[datafield]))
            data_types[datafield] = distribution
        elif datafields[datafield] == DistributionType.BETA:
            distribution = BetaDistribution()
            distribution.learn(list(df[datafield]))
            data_types[datafield] = distribution

    # Get the task_types
    task_types = df['Activity'].unique()
    if task_type_filter is not None:
        task_types = [tt for tt in task_types if task_type_filter(tt)]

    # Get the resources
    resources = df['Resource'].unique()

    # Mine the processing time distributions as they depend on other elements
    # For each task, add the tasks that preceded it as columns
    # tt_happened: task type -> []
    # each entry i in the list for task type tt (i.e. tt_happened[tt][i])
    # is the number of times tasks of type tt have happened in a case in dataframe df up to but not including row i
    tt_happened = dict()
    for tt in task_types:
        tt_happened[tt] = []
    current_case = None
    previous_task = None
    i = 0
    for index, row in df.iterrows():
        if row['Case ID'] != current_case:  # if we are starting a new case, set all task counts to 0 again
            current_case = row['Case ID']
            for tt in task_types:
                tt_happened[tt].append(0)
        else:  # if we are on a case, task counts remain the same, but increase for the previous task
            for tt in task_types:
                if tt == previous_task:
                    tt_happened[tt].append(tt_happened[tt][i-1] + 1)
                else:
                    tt_happened[tt].append(tt_happened[tt][i-1])
        previous_task = row['Activity']
        i += 1
    for tt in task_types:
        df[tt] = tt_happened[tt]
    # Now generate the distribution
    processing_times = StratifiedNumericDistribution()
    features = ['Activity', 'Resource'] + list(datafields.keys()) + list(task_types)
    onehot = ['Activity', 'Resource'] + [datafield for datafield in datafields if datafields[datafield] == DistributionType.CATEGORICAL]
    standardization = [datafield for datafield in datafields if datafields[datafield] != DistributionType.CATEGORICAL]
    processing_times.learn(df[features + ['Duration']], 'Duration', features, onehot, standardization, 'Activity')

    # Mine the control flow
    initial_tasks = dict()
    following_task = dict()
    interarrival_times = []
    last_arrival_time = None
    for index, row in df_cases.iterrows():
        if last_arrival_time is not None:
            interarrival_times.append((row['case_start'] - last_arrival_time).total_seconds()/3600)
        last_arrival_time = row['case_start']
        if not row['trace'][0] in initial_tasks.keys():
            initial_tasks[row['trace'][0]] = 0
        initial_tasks[row['trace'][0]] += 1
        for i in range(len(row['trace'])):
            predecessor = row['trace'][i]
            if i+1 >= len(row['trace']):
                successor = None
            else:
                successor = row['trace'][i+1]
            if not (predecessor, successor) in following_task:
                following_task[(predecessor, successor)] = 0
            following_task[(predecessor, successor)] += 1
    mean_interarrival_time = sum(interarrival_times)/len(interarrival_times)  # Assuming exponential distribution, so we only need the mean
    initial_task_distribution = []
    for it in initial_tasks:
        initial_task_distribution.append((initial_tasks[it]/len(df_cases), it))
    next_task_distribution = dict()
    task_occurrences = dict()
    for (predecessor, successor) in following_task:
        if predecessor not in next_task_distribution.keys():
            next_task_distribution[predecessor] = dict()
            task_occurrences[predecessor] = 0
        next_task_distribution[predecessor][successor] = following_task[(predecessor, successor)]
        task_occurrences[predecessor] += following_task[(predecessor, successor)]
    for predecessor in next_task_distribution:
        successors = []
        for successor in next_task_distribution[predecessor]:
            successors.append((next_task_distribution[predecessor][successor]/task_occurrences[predecessor], successor))
        next_task_distribution[predecessor] = successors

    # Mine the resource pools
    df_resources = df.groupby(['Activity', 'Resource'], as_index=False).agg(Duration_mean=('Duration', 'mean'), Duration_std=('Duration', 'std'), Resource_count=('Resource', 'count'))
    resource_pools = dict()
    for tt in task_types:
        resource_pools[tt] = []
    for index, row in df_resources.iterrows():
        if row["Resource_count"] > min_resource_count:
            resource_pools[row['Activity']].append(row['Resource'])

    # Mine the resource schedules
    begin = min(df['Start Timestamp'])
    end = max(df['Complete Timestamp'])
    hr = (begin, begin + resource_schedule_timeunit)
    schedule = [[] for i in range(resource_schedule_repeat)]
    resource_presence = dict()  # nr of hours during which a resource was present
    for r in resources:
        resource_presence[r] = 0
    x = 0
    while hr[1] <= end:
        # Tasks are within the hour hr (or other timeunit if that is chosen), if they hour ends or begins between the start and end of the task
        tasks_in_hour = df[((df['Start Timestamp'] <= hr[0]) & (df['Complete Timestamp'] >= hr[0])) | (
                (df['Start Timestamp'] <= hr[1]) & (df['Complete Timestamp'] >= hr[1]))]
        resources_in_hour = tasks_in_hour['Resource'].unique()
        for r in resources_in_hour:
            resource_presence[r] += 1
        schedule[x % resource_schedule_repeat].append(len(resources_in_hour))
        x += 1
        hr = (hr[0] + datetime.timedelta(hours=1), hr[1] + datetime.timedelta(hours=1))
    for x in range(resource_schedule_repeat):
        schedule[x] = round(mean(schedule[x]))
    resource_weights = []
    for r in resources:
        resource_weights.append(resource_presence[r])


log = pandas.read_parquet("../bpo/resources/BPI Challenge 2017 - clean Jan Feb.parquet")
res = mine_problem(log, datafields={'ApplicationType': DistributionType.CATEGORICAL, 'LoanGoal': DistributionType.CATEGORICAL, 'RequestedAmount': DistributionType.BETA})
