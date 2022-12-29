import matplotlib.pyplot as plt
import pandas


def boxplot(series):
    """
    Creates a boxplot for each element in the data series.
    The data series is a dictionary. Each key is used as the label of the boxplot.
    Each value is a list of numerical data that is used to construct the boxplot.

    :param series: a dictionary that maps a label to a list of numerical data.
    """
    plt.boxplot(series.values())
    plt.xticks(ticks=[i for i in range(1, len(series)+1)], labels=series.keys(), rotation=45)
    plt.show()


def line_with_ci(series):
    """
    Creates a line graph for the data series.
    The data series is a dictionary. Each key is a numerical value that represents an x-coordinate.
    Each value is a pair of numerical values, where the first element is the y-coordinate and the
    second element is an interval ci around the y-coordinate. A line graph is creates based on the
    (x, y) values with a bar around the y-ci, y+ci values. Typically, the ci value represents the
    confidence interval.

    :param series: a dictionary that maps numerical x values to (y, ci) numerical pairs.
    """
    x = series.keys()
    y = [mean for (mean, h) in series.values()]
    ci_bottom = [mean-h for (mean, h) in series.values()]
    ci_top = [mean+h for (mean, h) in series.values()]

    plt.plot(x, y)
    plt.fill_between(x, ci_bottom, ci_top, color='blue', alpha=0.1)
    plt.show()


def statistics(log, datetime_format="%Y/%m/%d %H:%M:%S"):
    """
    Creates statistics for the interarrival time and the processing times of the given log.
    Returns the statistics as a dictionary with the labels of the statistics as keys and as values lists
    with all the observed times. The log must contain the columns case_id, task, resource, start_time, completion_time.

    :param log: a pandas dataframe containing the log.
    :param datetime_format: optional parameter with the datetime formatting rule that will be used to interpret the start and completion timestamps
    """
    df = log.copy()
    df['start_time'] = pandas.to_datetime(df['start_time'], format=datetime_format)
    df['completion_time'] = pandas.to_datetime(df['completion_time'], format=datetime_format)
    df['duration'] = df[['start_time', 'completion_time']].apply(lambda tss: (tss[1] - tss[0]).total_seconds() / 3600, axis=1)

    df_cases = df.groupby('case_id').agg(case_start=('start_time', 'min'), case_complete=('start_time', 'min'), trace=('task', lambda tss: list(tss)))
    df_cases = df_cases.sort_values(by='case_start')

    task_types = df['task'].unique()

    resources = df['resource'].unique()

    interarrival_times = []
    last_arrival_time = None
    processing_times = dict()
    for tt in task_types:
        processing_times[tt] = list(df[df['task'] == tt]['duration'])
    for index, row in df_cases.iterrows():
        if last_arrival_time is not None:
            interarrival_times.append((row['case_start'] - last_arrival_time).total_seconds() / 3600)
        last_arrival_time = row['case_start']

    return {'Interarrrival times': interarrival_times, **processing_times}