import pandas
import datetime
from statistics import mean
from problems import MinedProblem
from distributions import DistributionType, CategoricalDistribution, BetaDistribution, GammaDistribution, NormalDistribution, StratifiedNumericDistribution


def mine_problem(log, task_type_filter=None, datetime_format="%Y/%m/%d %H:%M:%S", earliest_start=None, latest_completion=None, min_resource_count=2, resource_schedule_timeunit=datetime.timedelta(hours=1), resource_schedule_repeat=168, datafields=dict()):
    """
    Mines a problem and returns it as a :class:`.problems.Problem` that can be simulated.
    The log from which the model is mined must at least have the columns
    Case ID, Activity, Resource, Start Timestamp, Complete Timestamp,
    which identify the corresponding event log information. Activity labels
    are the same as Task Types for the purposes of the problem definition.
    The timing distributions associated with the problem are all in hours.
    Only cases that start on or after earliest_start and complete on or after latest_completion will be taken into account.
    Datafields specifies the columns in the log for which data fields will be learned. Datafields is a dictionary that maps column names to probability distributions.
    For the corresponding log column, a data type will be learned according to the specified distribution. The simulator can then draw samples for the distribution.

    :param log: a pandas dataframe from which the problem must be mined.
    :param task_type_filter: a function that takes the name of a task type/ activity
                             and returns if it should be included, or None to include all task types.
    :param datetime_format: the datetime format the Start Timestamp and Complete Timestamp columns use.
    :param earliest_start: a datetime object that, if not None, indicates that only cases that start on or after this datetime should be included. :param:`earliest_start` and :param:`latest_completion` should either both be None or both have a value.
    :param latest_completion: a datetime object that, if not None, indicates that only cases that complete on or before this datetime should be included. :param:`earliest_start` and :param:`latest_completion` should either both be None or both have a value.
    :param min_resource_count: the minimum number of times a resource must have executed a task
                               of a particular type, for it to be considered in the pool of resources for
                               the task type. This must be greater than 1, otherwise the standard deviation
                               of the processing time cannot be computed.
    :param resource_schedule_timeunit: the timeunit in which resource schedules should be represented. Default is 1 hour.
    :param resource_schedule_repeat: the number of times after which the resource schedule is expected to repeat itself. Default is 168 repeats (of 1 hour is a week).
    :return: a :class:`.problems.Problem`.
    :param datafields: a mapping of string to DistributionType, where string must be the name is one of the columns of the log.

    """

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
    schedule = [[] for _ in range(resource_schedule_repeat)]
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

    # CREATE THE PROBLEM
    result = MinedProblem()
    result.schedule = schedule
    result.resource_weights = resource_weights
    result.task_types = list(task_types)  # The task types
    result.resources = list(resources)  # The resources
    result.initial_task_distribution = initial_task_distribution  # The initial task type distribution
    result.next_task_distribution = next_task_distribution  # The next task type distribution per task type
    result.mean_interarrival_time = mean_interarrival_time  # The interarrival time
    result.resource_pools = resource_pools  # The resource pool per task type
    result.data_types = data_types
    result.processing_times = processing_times  # The processing time distributions

    return result
