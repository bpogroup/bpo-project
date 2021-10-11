import matplotlib.pyplot as plt


def boxplot(series):
    """
    Creates a boxplot for each element in the data series.
    The data series is a dictionary. Each key is used as the label of the boxplot.
    Each value is a list of numerical data that is used to construct the boxplot.

    :param series: a dictionary that maps a label to a list of numerical data.
    """
    plt.boxplot(series.values())
    plt.xticks(ticks=[i for i in range(1, len(series)+1)], labels=series.keys())
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
