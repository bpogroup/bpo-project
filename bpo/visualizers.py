import matplotlib.pyplot as plt


def boxplot(series):
    plt.boxplot(series.values())
    plt.xticks(ticks=[i for i in range(1, len(series)+1)], labels=series.keys())
    plt.show()


def line_with_ci(series):
    x = series.keys()
    y = [mean for (mean, h) in series.values()]
    ci_bottom = [mean-h for (mean, h) in series.values()]
    ci_top = [mean+h for (mean, h) in series.values()]

    plt.plot(x, y)
    plt.fill_between(x, ci_bottom, ci_top, color='blue', alpha=0.1)
    plt.show()
