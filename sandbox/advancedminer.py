import random

import pandas
import datetime
from statistics import mean
from enum import Enum, auto
import scipy


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

    # learn the datafields and corresponding disrtibutions
    datatypes = dict()
    for datafield in datafields:
        if datafields[datafield] == DistributionType.CATEGORICAL:
            distribution = CategoricalDistribution()
            distribution.learn(list(df[datafield].value_counts().index), list(df[datafield].value_counts().values))
            datatypes[datafield] = distribution
        elif datafields[datafield] == DistributionType.GAMMA:
            distribution = GammaDistribution()
            distribution.learn(list(df[datafield]))
            datatypes[datafield] = distribution
        elif datafields[datafield] == DistributionType.NORMAL:
            distribution = NormalDistribution()
            distribution.learn(list(df[datafield]))
            datatypes[datafield] = distribution
        elif datafields[datafield] == DistributionType.BETA:
            distribution = BetaDistribution()
            distribution.learn(list(df[datafield]))
            datatypes[datafield] = distribution

    return datatypes

    task_types = df['Activity'].unique()
    if task_type_filter is not None:
        task_types = [tt for tt in task_types if task_type_filter(tt)]

    resources = df['Resource'].unique()

    df['Duration'] = df[['Start Timestamp', 'Complete Timestamp']].apply(lambda tss: (tss[1]-tss[0]).total_seconds()/3600, axis=1)


    initial_tasks = dict()
    following_task = dict()
    interarrival_times = []
    last_arrival_time = None
    for index, row in df_cases.iterrows():
        if last_arrival_time is not None:
            interarrival_times.append((row['Start Timestamp'] - last_arrival_time).total_seconds()/3600)
        last_arrival_time = row['Start Timestamp']
        if not row['Trace'][0] in initial_tasks.keys():
            initial_tasks[row['Trace'][0]] = 0
        initial_tasks[row['Trace'][0]] += 1
        for i in range(len(row['Trace'])):
            predecessor = row['Trace'][i]
            if i+1 >= len(row['Trace']):
                successor = None
            else:
                successor = row['Trace'][i+1]
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
    df_resources = df.groupby(['Activity', 'Resource'], as_index=False).agg(Duration_mean=('Duration', 'mean'), Duration_std=('Duration', 'std'), Resource_count=('Resource', 'count'))
    resource_pools = dict()
    processing_time_distribution = dict()
    for tt in task_types:
        resource_pools[tt] = []
    for index, row in df_resources.iterrows():
        if row["Resource_count"] > min_resource_count:
            resource_pools[row['Activity']].append(row['Resource'])
            processing_time_distribution[(row['Activity'], row['Resource'])] = (row['Duration_mean'], row['Duration_std'])

    # MINE THE RESOURCE SCHEDULE
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
print(len(res))
