import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def distribution(title, xlabel, data):
    fig = plt.figure()
    ax = fig.gca(
            title=title,
            xlabel=xlabel,
            ylabel="Probability density"
            )
    for series_label, series_data in data.items():
        # Plot a filled kernel density estimate
        sns.distplot(series_data, label=series_label,
                ax=ax, hist=False, kde_kws={"shade": True})
    return fig
