# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome: Daniella Yuka Hirosue, Lara Oliveira Luzeiro, Renan Yugo Ueda
# RA: 813008, 813259, 813346
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from nltk.util import ngrams


# ==========================================================
# INFORMACOES GERAIS
# ==========================================================

def informacoes_gerais(df):

    print("\n" + "=" * 70)
    print("INFORMACOES GERAIS")
    print("=" * 70)

    print(f"Quantidade de documentos: {len(df)}")
    print(f"Quantidade de atributos: {df.shape[1]}")
    print(f"Quantidade de categorias: {df['Category'].nunique()}")

    print("\nValores nulos:")
    print(df.isnull().sum())


# ==========================================================
# DISTRIBUICAO DAS CLASSES
# ==========================================================

def distribuicao_classes(df):

    freq_classes = df["Category"].value_counts()

    print("\n" + "=" * 70)
    print("DISTRIBUICAO DAS CLASSES")
    print("=" * 70)

    print("\nDistribuicao das classes:")
    print(freq_classes)

    print("\nDistribuicao percentual das classes:")

    print(
        round(
            df["Category"].value_counts(normalize=True) * 100,
            2
        )
    )


# ==========================================================
# DOCUMENTOS DUPLICADOS
# ==========================================================

def analisar_duplicados(df):

    duplicados = df["Body"].duplicated().sum()

    print("\n" + "=" * 70)
    print("DOCUMENTOS DUPLICADOS")
    print("=" * 70)

    print(f"Quantidade de duplicados: {duplicados}")

    print(
        f"Percentual: "
        f"{100 * duplicados / len(df):.2f}%"
    )


# ==========================================================
# PROBLEMAS DE CODIFICACAO
# ==========================================================

def problemas_codificacao(df):

    padroes = [
        "Ã",
        "Â",
        "¤",
        "¢"
    ]

    quantidade = 0

    for texto in df["Body"].astype(str):

        if any(
            p in texto
            for p in padroes
        ):
            quantidade += 1

    print("\n" + "=" * 70)
    print("PROBLEMAS DE CODIFICACAO")
    print("=" * 70)

    print(
        f"Documentos afetados: "
        f"{quantidade}"
    )

    print(
        f"Percentual: "
        f"{100 * quantidade / len(df):.2f}%"
    )


# ==========================================================
# MEDIDAS DESCRITIVAS
# ==========================================================

def medidas_descritivas(df):

    df["word_count"] = df["Body"].astype(str).apply(
        lambda texto: len(texto.split())
    )

    print("\n" + "=" * 70)
    print("MEDIDAS DESCRITIVAS")
    print("=" * 70)

    print("\nEstatisticas do tamanho dos documentos:")
    print(df["word_count"].describe())

    print(f"\nMenor documento: {df['word_count'].min()} palavras")
    print(f"Maior documento: {df['word_count'].max()} palavras")
    print(f"Media de palavras por documento: {df['word_count'].mean():.2f}")


# ==========================================================
# TAMANHO DOS DOCUMENTOS POR CLASSE
# ==========================================================

def tamanho_por_categoria(df):

    if "word_count" not in df.columns:

        df["word_count"] = (
            df["Body"]
            .astype(str)
            .apply(lambda x: len(x.split()))
        )

    print("\n" + "=" * 70)
    print("TAMANHO DOS DOCUMENTOS POR CLASSE")
    print("=" * 70)

    print(
        df.groupby("Category")["word_count"]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std"
        ])
    )


# ==========================================================
# REFERENCIAS LEGISLATIVAS
# ==========================================================

def analisar_referencias_legais(df):

    padrao = r"(ARTIGO_\d+|LEI_\d+|DECRETO_\d+)"

    referencias = (
        df["Body"]
        .astype(str)
        .str.count(padrao)
    )

    print("\n" + "=" * 70)
    print("REFERENCIAS LEGISLATIVAS")
    print("=" * 70)

    print(referencias.describe())


# ==========================================================
# ANALISE LEXICAL
# ==========================================================

def fix_mojibake(text):

    try:
        return text.encode("latin-1").decode("utf-8")
    except:
        return text


def tokenize(text):

    stop_words = {
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em',
        'um', 'para', 'é', 'com', 'não', 'uma',
        'os', 'no'
    }

    text = text.lower()

    words = re.sub(
        r'[^a-záéíóúâêôãõç]',
        ' ',
        text
    ).split()

    return [
        w for w in words
        if w not in stop_words and len(w) > 2
    ]


def analise_lexical(df):

    df["Body_Clean"] = df["Body"].astype(str).apply(
        fix_mojibake
    )

    sample_texts = df["Body_Clean"].sample(
        min(5000, len(df)),
        random_state=42
    )

    all_words = [
        word
        for text in sample_texts
        for word in tokenize(text)
    ]

    vocabulario = set(all_words)

    diversidade_lexical = (
        len(vocabulario) / len(all_words)
        if len(all_words) > 0
        else 0
    )

    common_words = Counter(all_words).most_common(15)

    print("\n" + "=" * 70)
    print("ANALISE LEXICAL")
    print("=" * 70)

    print(f"Tamanho do vocabulario: {len(vocabulario)}")
    print(f"Diversidade lexical: {diversidade_lexical:.4f}")

    print("\nTop 15 palavras mais frequentes:")

    for palavra, freq in common_words:
        print(f"{palavra}: {freq}")

    return common_words


# ==========================================================
# DIVERSIDADE LEXICAL POR CLASSE
# ==========================================================

def diversidade_por_categoria(df):

    print("\n" + "=" * 70)
    print("DIVERSIDADE LEXICAL POR CLASSE")
    print("=" * 70)

    for categoria in sorted(
        df["Category"].unique()
    ):

        textos = df[
            df["Category"] == categoria
        ]["Body"]

        palavras = []

        for texto in textos:

            palavras.extend(
                tokenize(
                    fix_mojibake(str(texto))
                )
            )

        vocabulario = len(
            set(palavras)
        )

        diversidade = (
            vocabulario / len(palavras)
            if len(palavras) > 0
            else 0
        )

        print(
            f"Classe {categoria}: "
            f"{diversidade:.4f}"
        )


# ==========================================================
# PALAVRAS MAIS FREQUENTES POR CLASSE
# ==========================================================

def palavras_por_categoria(df):

    print("\n" + "=" * 70)
    print("PALAVRAS MAIS FREQUENTES POR CLASSE")
    print("=" * 70)

    for categoria in sorted(
        df["Category"].unique()
    ):

        textos = df[
            df["Category"] == categoria
        ]["Body"]

        palavras = []

        for texto in textos:

            palavras.extend(
                tokenize(
                    fix_mojibake(str(texto))
                )
            )

        top = Counter(
            palavras
        ).most_common(10)

        print(
            f"\nClasse {categoria}"
        )

        for palavra, freq in top:

            print(
                f"{palavra}: {freq}"
            )


# ==========================================================
# BIGRAMAS FREQUENTES
# ==========================================================

def bigramas_frequentes(df):

    palavras = []

    sample_texts = df["Body"].sample(
        min(5000, len(df)),
        random_state=42
    )

    for texto in sample_texts:

        palavras.extend(
            tokenize(
                fix_mojibake(str(texto))
            )
        )

    top_bigramas = Counter(
        ngrams(
            palavras,
            2
        )
    ).most_common(15)

    print("\n" + "=" * 70)
    print("TOP 15 BIGRAMAS")
    print("=" * 70)

    for bg, freq in top_bigramas:

        print(
            f"{bg[0]} {bg[1]}: {freq}"
        )


# ==========================================================
# GRAFICO 1 - DISTRIBUICAO DAS CLASSES
# ==========================================================

def grafico_distribuicao_classes(df):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x="Category",
        order=df["Category"].value_counts().index,
        hue="Category",
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_title("Distribuicao das Classes")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Frequencia")

    plt.tight_layout()
    plt.show()


# ==========================================================
# GRAFICO 2 - BOXPLOT
# ==========================================================

def boxplot_tamanho_textos(df):

    if "word_count" not in df.columns:
        df["word_count"] = df["Body"].astype(str).apply(
            lambda texto: len(texto.split())
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="Category",
        y="word_count",
        hue="Category",
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Distribuicao do Numero de Palavras por Classe"
    )

    ax.set_ylim(
        0,
        df["word_count"].quantile(0.95)
    )

    ax.set_xlabel("Categoria")
    ax.set_ylabel("Numero de Palavras")

    plt.tight_layout()
    plt.show()


# ==========================================================
# GRAFICO 3 - HISTOGRAMA
# ==========================================================

def histograma_tamanho_textos(df):

    if "word_count" not in df.columns:
        df["word_count"] = df["Body"].astype(str).apply(
            lambda texto: len(texto.split())
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="word_count",
        bins=50,
        kde=True,
        ax=ax
    )

    ax.set_title(
        "Distribuicao do Tamanho dos Documentos"
    )

    ax.set_xlabel("Numero de Palavras")
    ax.set_ylabel("Frequencia")

    plt.tight_layout()
    plt.show()


# ==========================================================
# GRAFICO 4 - REFERENCIAS LEGISLATIVAS
# ==========================================================

def grafico_referencias_legais(df):

    padrao = r"(ARTIGO_\d+|LEI_\d+|DECRETO_\d+)"

    refs = (
        df["Body"]
        .astype(str)
        .str.count(padrao)
    )

    plt.figure(figsize=(10, 5))

    sns.histplot(
        refs,
        bins=40,
        kde=True
    )

    plt.title(
        "Quantidade de Referencias Legislativas"
    )

    plt.xlabel(
        "Numero de referencias"
    )

    plt.ylabel(
        "Frequencia"
    )

    plt.tight_layout()

    plt.show()


# ==========================================================
# GRAFICO 5 - PALAVRAS FREQUENTES
# ==========================================================

def palavras_frequentes(common_words):

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=[w[1] for w in common_words],
        y=[w[0] for w in common_words],
        hue=[w[0] for w in common_words],
        palette="mako",
        legend=False,
        ax=ax
    )

    ax.set_title(
        "Top 15 Palavras Mais Frequentes"
    )

    ax.set_xlabel("Frequencia")
    ax.set_ylabel("Palavra")

    plt.tight_layout()
    plt.show()