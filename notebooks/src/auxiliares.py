import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    kruskal,
    levene,
    mannwhitneyu,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)


def tabela_distribuicao_frequencias(dataframe, coluna, coluna_frequencia=False):
    """Cria uma tabela de distribuição de frequências para uma coluna de um dataframe.
    Espera uma coluna categórica.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe com os dados.
    coluna : str
        Nome da coluna categórica.
    coluna_frequencia : bool
        Informa se a coluna passada já é com os valores de frequência ou não. Padrão: False

    Returns
    -------
    pd.DataFrame
        Dataframe com a tabela de distribuição de frequências.
    """

    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica["frequencia"] = dataframe[coluna]
        df_estatistica["frequencia_relativa"] = (
            df_estatistica["frequencia"] / df_estatistica["frequencia"].sum()
        )
    else:
        df_estatistica["frequencia"] = dataframe[coluna].value_counts().sort_index()
        df_estatistica["frequencia_relativa"] = (
            dataframe[coluna].value_counts(normalize=True).sort_index()
        )

    df_estatistica["frequencia_acumulada"] = df_estatistica["frequencia"].cumsum()
    df_estatistica["frequencia_relativa_acumulada"] = df_estatistica[
        "frequencia_relativa"
    ].cumsum()

    return df_estatistica





#Perform the Shapiro-Wilk test for normality.
#The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
def analise_shapiro(dataframe, alfa=0.05):
    print("Teste de Shapiro-Wilk")
    for coluna in dataframe.columns:
        estatistica_sw, valor_p_sw = shapiro(dataframe[coluna], nan_policy="omit")
        print(f"{estatistica_sw=:.3f}")
        if valor_p_sw > alfa:
            print(f"{coluna} segue uma distribuição normal (valor p: {valor_p_sw:.3f})")
        else:
            print(
                f"{coluna} não segue uma distribuição normal (valor p: {valor_p_sw:.3f})"
            )

#Perform Levene test for equal variances.
#The Levene test tests the null hypothesis that all input samples are from populations with equal variances. Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.

def analise_levene(dataframe, alfa=0.05, centro="mean"):
    print("Teste de Levene")

    estatistica_levene, valor_p_levene = levene(
        *[dataframe[coluna] for coluna in dataframe.columns],
        center=centro,
        nan_policy="omit",
    )

    print(f"{estatistica_levene=:.3f}")
    if valor_p_levene > alfa:
        print(f"Variâncias iguais (valor p: {valor_p_levene:.3f})")
    else:
        print(f"Ao menos uma variância é diferente (valor p: {valor_p_levene:.3f})")

#Perform the Shapiro-Wilk test for normality and Levene test for equal variances.
def analises_shapiro_levene(dataframe, alfa=0.05, centro="mean"):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, centro)

#Calculate the T-test for the means of two independent samples of scores.
#This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.
    
def analise_ttest_ind(
    dataframe,
    alfa=0.05,
    variancias_iguais=True,
    alternativa="two-sided",
):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy="omit",
    )

    print(f"{estatistica_ttest=:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")

#Calculate the t-test on TWO RELATED samples of scores, a and b.
#This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
def analise_ttest_rel(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns],
        alternative=alternativa,
        nan_policy="omit",
    )

    print(f"{estatistica_ttest=:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")

#Perform one-way ANOVA.
#The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.
def analise_anova_one_way(
    dataframe,
    alfa=0.05,
):

    print("Teste ANOVA one way")
    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns], nan_policy="omit"
    )

    print(f"{estatistica_f=:.3f}")
    if valor_p_f > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_f:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_f:.3f})")

#Calculate the Wilcoxon signed-rank test.
#The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
def analise_wilcoxon(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):

    print("Teste de Wilcoxon")
    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa,
    )

    print(f"{estatistica_wilcoxon=:.3f}")
    if valor_p_wilcoxon > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_wilcoxon:.3f})")

#Perform the Mann-Whitney U rank test on two independent samples.
#The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is the same as the distribution underlying sample y. It is often used as a test of difference in location between distributions.
def analise_mannwhitneyu(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):

    print("Teste de Mann-Whitney")
    estatistica_mw, valor_p_mw = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa,
    )

    print(f"{estatistica_mw=:.3f}")
    if valor_p_mw > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")

#Compute the Friedman test for repeated samples.
#The Friedman test tests the null hypothesis that repeated samples of the same individuals have the same distribution. It is often used to test for consistency among samples obtained in different ways. For example, if two sampling techniques are used on the same set of individuals, the Friedman test can be used to determine if the two sampling techniques are consistent.
def analise_friedman(
    dataframe,
    alfa=0.05,
):

    print("Teste de Friedman")
    estatistica_friedman, valor_p_friedman = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )

    print(f"{estatistica_friedman=:.3f}")
    if valor_p_friedman > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_friedman:.3f})")

#Compute the Kruskal-Wallis H-test for independent samples.
#The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs. Post hoc comparisons between groups are required to determine which groups are different.
def analise_kruskal(
    dataframe,
    alfa=0.05,
):

    print("Teste de Kruskal")
    estatistica_kruskal, valor_p_kruskal = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )

    print(f"{estatistica_kruskal=:.3f}")
    if valor_p_kruskal > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_kruskal:.3f})")


def remove_outliers(dados, largura_bigodes=1.5):
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    return dados[(dados >= q1 - largura_bigodes * iqr) & (dados <= q3 + largura_bigodes * iqr)]
