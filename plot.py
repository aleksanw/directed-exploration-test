import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca()

def distribution(path, title, xlabel, data):
    """Not thread-safe."""
    ax.cla()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_ylim([0,3])
    ax.set_xlim([-10,10])

    for series_label, series_data in data.items():
        # Plot a filled kernel density estimate
        sns.distplot(series_data, label=series_label,
                ax=ax, hist=False, kde_kws={"shade": True})

    fig.savefig(path)
