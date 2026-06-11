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
from collections import Counter
import re


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
# GRAFICO 4 - PALAVRAS FREQUENTES
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