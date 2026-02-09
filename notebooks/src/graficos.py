import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def composicao_histograma_boxplot(dataframe, coluna, intervalos="auto"):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": (0.15, 0.85), "hspace": 0.02},
    )

    sns.boxplot(
        data=dataframe,
        x=coluna,
        showmeans=True,
        meanline=True,
        meanprops={"color": "C1", "linewidth": 1.5, "linestyle": "--"},
        medianprops={"color": "C2", "linewidth": 1.5, "linestyle": "--"},
        ax=ax1,
    )

    sns.histplot(data=dataframe, x=coluna, kde=True, bins=intervalos, ax=ax2)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", color="gray", alpha=0.5)
        ax.set_axisbelow(True)

    ax2.axvline(dataframe[coluna].mean(), color="C1", linestyle="--", label="Média")
    ax2.axvline(dataframe[coluna].median(), color="C2", linestyle="--", label="Mediana")
    ax2.axvline(dataframe[coluna].mode()[0], color="C3", linestyle="--", label="Moda")

    ax2.legend()

    plt.show()