# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocoba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome:
# RA:
# ################################################################

from __future__ import annotations

import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paleta de cores para as classes
MAPEAMENTO_CLASSES = {
    0: "Acordao",
    1: "ARE",
    2: "Despacho",
    3: "RE",
    4: "Sentenca",
}

CORES_CLASSES = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

# Stopwords para analise exploratoria (sem dependencia de NLTK)
STOPWORDS_BASICAS = {
    "que", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "um", "uma", "uns", "umas", "ao", "aos", "pela",
    "pelo", "pelos", "pelas", "este", "esta", "estes", "estas", "esse", "essa",
    "esses", "essas", "isso", "isto", "aquele", "aquela", "aqueles", "aquelas",
    "ser", "ter", "se", "ou", "mas", "pois", "porque", "quando", "como",
    "mais", "muito", "todo", "toda", "todos", "todas", "seu", "sua", "seus", "suas",
    "meu", "minha", "nosso", "nossa", "nao", "nem", "foi", "eram", "sera", "seria",
    "assim", "ainda", "agora", "tambem", "ja", "ate", "desde", "pode", "podem",
    "deve", "devem", "sendo", "artigo", "artigos", "lei", "inc", "par",
    "sobre", "contra", "entre", "apos", "ante", "sob", "perante",
}


# ==============================================================
# FUNCOES INICIAIS
# ==============================================================

def resolver_caminho_arquivo(nome_arquivo: str) -> Path:
    """
    Resolve o caminho de um arquivo procurando no diretorio atual
    e na subpasta 'dataset/', nesta ordem.
    """
    caminho_direto = Path(nome_arquivo)
    caminho_dataset = Path("dataset") / nome_arquivo

    if caminho_direto.exists():
        return caminho_direto
    if caminho_dataset.exists():
        return caminho_dataset

    raise FileNotFoundError(
        f"Arquivo '{nome_arquivo}' nao encontrado nem no diretorio atual nem em 'dataset/'."
    )


def carregar_csv(nome_arquivo: str) -> pd.DataFrame:
    """Carrega um arquivo CSV, procurando em 'dataset/' se necessario."""
    caminho = resolver_caminho_arquivo(nome_arquivo)
    return pd.read_csv(caminho, engine="python")


def carregar_dados(
    arquivo_treino: str = "train.csv",
    arquivo_teste: str = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os arquivos de treino e teste.
    Aceita paths relativos (resolve automaticamente dataset/).
    Garante tipos numpy puros para compatibilidade com scikit-learn.
    """
    dados_treino = carregar_csv(arquivo_treino)
    dados_teste = carregar_csv(arquivo_teste)
    dados_treino["Body"] = dados_treino["Body"].astype(str)
    if "Category" in dados_treino.columns:
        dados_treino["Category"] = dados_treino["Category"].astype(np.int64)
    dados_teste["Body"] = dados_teste["Body"].astype(str)
    return dados_treino, dados_teste


def obter_mapeamento_classes() -> dict[int, str]:
    """Retorna copia do mapeamento de inteiros para nomes de classes."""
    return MAPEAMENTO_CLASSES.copy()


# ==============================================================
# FILTRAGEM DE ROTULOS VALIDOS/INVALIDOS
# ==============================================================

def filtrar_rotulos_validos(
    dados: pd.DataFrame,
    coluna_alvo: str = "Category",
) -> pd.DataFrame:
    """Retorna apenas linhas com Category em {0, 1, 2, 3, 4}."""
    rotulos_validos = set(MAPEAMENTO_CLASSES)
    return dados.loc[dados[coluna_alvo].isin(rotulos_validos)].copy()


# ==============================================================
# VISAO GERAL
# ==============================================================

def exibir_info_geral(df_treino: pd.DataFrame, df_teste: pd.DataFrame) -> None:
    """
    Exibe informacoes gerais dos datasets: shape, colunas e valores ausentes.
    Identifica amostras sem rotulo (Category == -1).
    """
    print("=" * 60)
    print("VISAO GERAL DOS DADOS")
    print("=" * 60)
    print(f"\nTreino : {df_treino.shape[0]:>6,} amostras | {df_treino.shape[1]} colunas")
    print(f"Teste  : {df_teste.shape[0]:>6,} amostras | {df_teste.shape[1]} colunas")
    print(f"\nColunas treino: {list(df_treino.columns)}")
    print(f"Colunas teste : {list(df_teste.columns)}")
    print("\nValores ausentes (treino):")
    print(df_treino.isnull().sum().to_string())
    if "Category" in df_treino.columns:
        sem_rotulo = (df_treino["Category"] == -1).sum()
        com_rotulo = (df_treino["Category"] >= 0).sum()
        print(f"\nAmostras sem rotulo (Category == -1) : {sem_rotulo:,}")
        print(f"Amostras rotuladas  (Category >= 0)  : {com_rotulo:,}")


# ==============================================================
# DOCUMENTOS DUPLICADOS
# ==============================================================

def analisar_duplicados(
    dados: pd.DataFrame,
    coluna_texto: str = "Body"
) -> int:
    """
    Analisa e exibe a quantidade de textos duplicados no dataset.
    Retorna o numero total de duplicados encontrados.
    """
    duplicados = dados[coluna_texto].duplicated().sum()
    total = len(dados)
    
    print("\n" + "=" * 60)
    print("DOCUMENTOS DUPLICADOS")
    print("=" * 60)
    print(f"Quantidade de duplicados : {duplicados:>8,}")
    if total > 0:
        print(f"Percentual do dataset    : {100 * duplicados / total:.2f}%")
        
    return int(duplicados)

# ==========================================================
# PROBLEMAS DE CODIFICACAO
# ==========================================================

def problemas_codificacao(
    dados: pd.DataFrame,
    coluna_texto: str = "Body"
) -> int:
    """
    Analisa a quantidade de documentos que apresentam problemas de codificacao
    (caracteres estranhos gerados por erro de encoding, como Ã, Â, ¤, ¢).
    Retorna o numero total de documentos afetados.
    """
    # Usando o operador | (OU) para buscar qualquer um dos padrões de uma vez
    padrao_regex = "Ã|Â|¤|¢"
    
    # Busca vetorizada: muito mais rápida que um laço for
    mascara = dados[coluna_texto].fillna("").astype(str).str.contains(padrao_regex)
    quantidade = mascara.sum()
    total = len(dados)
    
    print("\n" + "=" * 60)
    print("PROBLEMAS DE CODIFICACAO")
    print("=" * 60)
    print(f"Documentos afetados : {quantidade:>8,}")
    
    if total > 0:
        print(f"Percentual do dataset: {100 * quantidade / total:.2f}%")


# ==============================================================
# DISTRIBUICAO DE CLASSES
# ==============================================================

def calcular_distribuicao_classes(
    dados: pd.DataFrame,
    coluna_alvo: str = "Category",
    incluir_rotulos_invalidos: bool = True,
) -> pd.DataFrame:
    """
    Calcula a distribuicao das classes (contagem e percentual).
    Inclui rotulos invalidos na contagem se `incluir_rotulos_invalidos=True`.
    """
    distribuicao = (
        dados[coluna_alvo]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("classe")
        .reset_index(name="quantidade")
    )
    distribuicao["percentual"] = (
        distribuicao["quantidade"] / distribuicao["quantidade"].sum()
    ) * 100.0
    distribuicao["nome_classe"] = distribuicao["classe"].map(MAPEAMENTO_CLASSES)
    distribuicao["nome_classe"] = distribuicao["nome_classe"].fillna("Rotulo invalido")

    if incluir_rotulos_invalidos:
        return distribuicao
    return distribuicao.loc[distribuicao["classe"].isin(MAPEAMENTO_CLASSES)].copy()


def plotar_distribuicao_classes(
    distribuicao: pd.DataFrame,
    titulo: str = "Distribuicao das Classes",
    salvar_em: str | None = "figs/distribuicao_classes.png",
) -> None:
    """
    Plota distribuicao de classes com grafico de barras (contagem) e
    grafico de pizza (proporcao), exibindo apenas as classes validas.
    Valores de contagem sao exibidos acima de cada barra.
    """
    os.makedirs("figs", exist_ok=True)
    dist_valida = distribuicao[distribuicao["nome_classe"] != "Rotulo invalido"].copy()

    nomes = dist_valida["nome_classe"].tolist()
    contagens = dist_valida["quantidade"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico de barras com anotacoes
    bars = axes[0].bar(nomes, contagens, color=CORES_CLASSES[: len(nomes)], edgecolor="white")
    axes[0].set_title(f"{titulo} (Contagem)", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Numero de Amostras")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, contagens):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(contagens) * 0.01,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Grafico de pizza
    pct = [c / sum(contagens) * 100 for c in contagens]
    axes[1].pie(
        pct,
        labels=nomes,
        colors=CORES_CLASSES[: len(nomes)],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    axes[1].set_title(f"{titulo} (%)", fontweight="bold", fontsize=13)

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# COMPRIMENTO DOS TEXTOS
# ==============================================================

def extrair_texto_bruto(texto: str) -> str:
    """
    Extrai o conteudo textual do campo Body.
    Alguns registros estao no formato JSON: {"...texto..."} -> 'texto'.
    """
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    if texto.startswith("{") and texto.endswith("}"):
        texto = texto[1:-1]
    texto = texto.strip('"')
    return texto


def contar_tokens(texto: str) -> int:
    """Conta o numero de tokens (palavras) de um texto."""
    if pd.isna(texto):
        return 0
    return len(str(texto).split())


def calcular_estatisticas_tamanho_textos(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
) -> pd.DataFrame:
    """Retorna estatisticas gerais (media, mediana, desvio, min, max, percentis) de tamanho."""
    serie = dados[coluna_texto].fillna("").astype(str).apply(
        lambda t: contar_tokens(extrair_texto_bruto(t))
    )
    return pd.DataFrame([{
        "quantidade_documentos": int(serie.shape[0]),
        "media_tokens": float(serie.mean()),
        "mediana_tokens": float(serie.median()),
        "desvio_padrao_tokens": float(serie.std(ddof=0)),
        "minimo_tokens": int(serie.min()),
        "maximo_tokens": int(serie.max()),
        "percentil_90_tokens": float(serie.quantile(0.90)),
        "percentil_95_tokens": float(serie.quantile(0.95)),
    }])


def calcular_estatisticas_tamanho_por_classe(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
    coluna_alvo: str = "Category",
) -> pd.DataFrame:
    """
    Retorna estatisticas de tamanho dos textos por classe.
    Apenas para amostras com rotulos validos (0-4).
    """
    dados_aux = filtrar_rotulos_validos(dados, coluna_alvo=coluna_alvo).copy()
    dados_aux["quantidade_tokens"] = (
        dados_aux[coluna_texto]
        .fillna("")
        .astype(str)
        .apply(lambda t: contar_tokens(extrair_texto_bruto(t)))
    )
    estatisticas = (
        dados_aux.groupby(coluna_alvo)["quantidade_tokens"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    estatisticas["nome_classe"] = estatisticas[coluna_alvo].map(MAPEAMENTO_CLASSES)
    return estatisticas.rename(columns={
        "count": "quantidade_documentos",
        "mean": "media_tokens",
        "median": "mediana_tokens",
        "std": "desvio_padrao_tokens",
        "min": "minimo_tokens",
        "max": "maximo_tokens",
    })


def plotar_comprimento_textos(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
    coluna_alvo: str = "Category",
    salvar_em: str | None = "figs/comprimento_textos.png",
) -> None:
    """
    Plota boxplot e histograma do comprimento dos textos por classe.
    - Boxplot: revela outliers e distribuicao quartil por classe.
    - Histograma sobrepostos: mostra a distribuicao completa por classe.
    """
    os.makedirs("figs", exist_ok=True)
    dados_aux = filtrar_rotulos_validos(dados, coluna_alvo=coluna_alvo).copy()
    dados_aux["num_tokens"] = dados_aux[coluna_texto].fillna("").astype(str).apply(
        lambda t: contar_tokens(extrair_texto_bruto(t))
    )
    dados_aux["nome_classe"] = dados_aux[coluna_alvo].map(MAPEAMENTO_CLASSES)
    ordem = list(MAPEAMENTO_CLASSES.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(
        data=dados_aux, x="nome_classe", y="num_tokens",
        order=ordem, palette=CORES_CLASSES, ax=axes[0],
    )
    axes[0].set_title("Comprimento por Classe (Tokens)", fontweight="bold")
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Numero de Tokens")
    axes[0].tick_params(axis="x", rotation=20)

    for i, (idx, nome) in enumerate(MAPEAMENTO_CLASSES.items()):
        dados_classe = dados_aux[dados_aux[coluna_alvo] == idx]["num_tokens"]
        axes[1].hist(dados_classe, bins=40, alpha=0.55, label=nome, color=CORES_CLASSES[i])
    axes[1].set_title("Distribuicao do Comprimento", fontweight="bold")
    axes[1].set_xlabel("Numero de Tokens")
    axes[1].set_ylabel("Frequencia")
    axes[1].legend()

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# PALAVRAS FREQUENTES
# ==============================================================

def _tokenizar_texto_simples(
    texto: str,
    stopwords_personalizadas=None,
) -> list[str]:

    texto_bruto = extrair_texto_bruto(texto).lower()

    stopwords_ajustadas = STOPWORDS_BASICAS.copy()
    if stopwords_personalizadas is not None:
        stopwords_ajustadas.update(stopwords_personalizadas)

    # 1. Separar o texto por ESPAÇOS reais. 
    # Assim, "previdenciÃ¡rios" é lido como um bloco único.
    tokens_brutos = texto_bruto.split()
    
    # Lista de pontuações para limpar as bordas das palavras
    pontuacao = string.punctuation + "“”‘’…"

    tokens = []
    for token in tokens_brutos:
        # 2. Removemos vírgulas, pontos e aspas grudadas na palavra
        token_limpo = token.strip(pontuacao)
        
        # 3. Verificamos se tem 3 letras ou mais e não é stopword
        if len(token_limpo) >= 3 and token_limpo not in stopwords_ajustadas:
            
            # 4. A MÁGICA ESTÁ AQUI: 
            # Só aceitamos a palavra se ela for composta APENAS por letras normais.
            # Se ela tiver algum símbolo estranho de encoding (como Ã, ¡, ³, £), 
            # ela é sumariamente descartada da estatística.
            if re.match(r"^[a-zà-ÿ]+$", token_limpo):
                tokens.append(token_limpo)

    return tokens

def calcular_palavras_frequentes_por_classe(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
    coluna_alvo: str = "Category",
    top_n: int = 20,
    stopwords_personalizadas: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula as palavras mais frequentes por classe.
    Retorna DataFrame com colunas: classe, nome_classe, palavra, frequencia.
    """
    dados_validos = filtrar_rotulos_validos(dados, coluna_alvo=coluna_alvo)
    registros = []

    for classe, grupo in dados_validos.groupby(coluna_alvo):
        contador: Counter = Counter()
        for texto in grupo[coluna_texto].fillna("").astype(str):
            tokens = _tokenizar_texto_simples(texto, stopwords_personalizadas)
            contador.update(tokens)
        for palavra, frequencia in contador.most_common(top_n):
            registros.append({
                "classe": int(classe),
                "nome_classe": MAPEAMENTO_CLASSES.get(int(classe), "Rotulo invalido"),
                "palavra": palavra,
                "frequencia": int(frequencia),
            })

    return pd.DataFrame(registros)


def converter_para_dict_frequencias(
    df_freq: pd.DataFrame,
    n_palavras: int = 15,
) -> dict[str, list[tuple[str, int]]]:
    """
    Converte o DataFrame de palavras frequentes para um dicionario:
    {nome_classe: [(palavra, freq), ...]} mantendo a ordem numerica das classes.
    """
    resultado: dict[str, list[tuple[str, int]]] = {}
    
    # Agrupa pela coluna numérica 'classe' para garantir a ordem (0, 1, 2...)
    for classe_id, grupo in df_freq.groupby("classe"):
        # Pega o nome da classe correspondente
        nome_classe = grupo["nome_classe"].iloc[0]
        
        resultado[nome_classe] = list(
            zip(grupo["palavra"].tolist(), grupo["frequencia"].tolist())
        )[:n_palavras]
        
    return resultado


def plotar_palavras_frequentes(
    frequencias: dict[str, list[tuple[str, int]]],
    n_palavras: int = 15,
    salvar_em: str | None = "figs/palavras_frequentes.png",
) -> None:
    """
    Plota graficos de barras horizontais com as palavras mais frequentes por classe.
    Cada subplot corresponde a uma classe do dataset juridico.
    """
    os.makedirs("figs", exist_ok=True)
    n = len(frequencias)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))

    nomes_classes = list(MAPEAMENTO_CLASSES.values())
    for i, (nome_classe, palavras_freq) in enumerate(frequencias.items()):
        palavras = [p for p, _ in palavras_freq[:n_palavras]]
        contagens = [c for _, c in palavras_freq[:n_palavras]]
        idx_cor = nomes_classes.index(nome_classe) if nome_classe in nomes_classes else i
        cor = CORES_CLASSES[idx_cor % len(CORES_CLASSES)]
        axes[i].barh(palavras[::-1], contagens[::-1], color=cor, alpha=0.85)
        axes[i].set_title(nome_classe, fontweight="bold", fontsize=11)
        axes[i].set_xlabel("Frequencia")
        if i == 0:
            axes[i].set_ylabel("Palavras")

    plt.suptitle("Palavras Mais Frequentes por Classe", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# NUVEM DE PALAVRAS
# ==============================================================

def plotar_nuvem_palavras(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
    coluna_alvo: str = "Category",
    salvar_em: str | None = "figs/nuvem_palavras.png",
    stopwords_personalizadas: Iterable[str] | None = None,
) -> None:
    """
    Plota nuvens de palavras por classe juridica.
    Garante sincronia com as palavras frequentes calculadas, 
    utilizando as mesmas regras de tokenizacao.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud nao instalado. Execute: pip install wordcloud")
        return

    os.makedirs("figs", exist_ok=True)
    
    # 1. Utilizamos a nossa função oficial para contar as palavras corretamente!
    # Pegamos o top 80 para a nuvem ficar cheia
    df_freq = calcular_palavras_frequentes_por_classe(
        dados, 
        coluna_texto=coluna_texto, 
        coluna_alvo=coluna_alvo, 
        top_n=80, 
        stopwords_personalizadas=stopwords_personalizadas
    )

    fig, axes = plt.subplots(1, len(MAPEAMENTO_CLASSES), figsize=(5 * len(MAPEAMENTO_CLASSES), 4))
    
    # 2. Iteramos mantendo a ordem numérica das classes (0, 1, 2...)
    for i, (classe_id, nome_classe) in enumerate(MAPEAMENTO_CLASSES.items()):
        
        # Filtra as frequencias apenas da classe atual
        freq_classe = df_freq[df_freq["classe"] == classe_id]
        
        # Converte para um dicionário no formato {palavra: quantidade} que o WordCloud exige
        dict_freq = dict(zip(freq_classe["palavra"], freq_classe["frequencia"]))
        
        # Se o dicionário não estiver vazio, desenhamos a nuvem
        if dict_freq:
            wc = WordCloud(
                width=400,
                height=300,
                background_color="white",
                max_words=80,
                colormap="viridis",
            ).generate_from_frequencies(dict_freq) # <-- A magica acontece aqui
            
            axes[i].imshow(wc, interpolation="bilinear")
            
        axes[i].axis("off")
        axes[i].set_title(nome_classe, fontweight="bold", fontsize=12)

    plt.suptitle("Nuvens de Palavras por Classe", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# VOCABULARIO GLOBAL
# ==============================================================

def analisar_vocabulario(
    dados: pd.DataFrame,
    coluna_texto: str = "Body",
    coluna_alvo: str = "Category",
) -> Counter:
    """
    Analisa o vocabulario global do corpus rotulado.
    Exibe tipos unicos, total de tokens e os 20 termos mais frequentes.
    Util para dimensionar o vocabulario e identificar termos juridicos centrais.
    """
    dados_validos = filtrar_rotulos_validos(dados, coluna_alvo=coluna_alvo).copy()
    dados_validos["texto_bruto"] = dados_validos[coluna_texto].apply(extrair_texto_bruto)

    vocab_global: Counter = Counter()
    for texto in dados_validos["texto_bruto"]:
        for token in _tokenizar_texto_simples(texto):
            vocab_global[token] += 1

    total_tokens = sum(vocab_global.values())
    print("\nAnalise de Vocabulario (corpus rotulado):")
    print(f"  Tipos unicos (vocab total) : {len(vocab_global):>8,}")
    print(f"  Total de tokens            : {total_tokens:>8,}")
    print(f"  Tokens com freq >= 5       : {sum(1 for v in vocab_global.values() if v >= 5):>8,}")
    print(f"  Tokens com freq >= 10      : {sum(1 for v in vocab_global.values() if v >= 10):>8,}")
    print("\nTop 20 palavras mais frequentes (geral):")
    for palavra, freq in vocab_global.most_common(20):
        print(f"  {palavra:<30} {freq:>8,}")

    return vocab_global



__all__ = [
    "MAPEAMENTO_CLASSES",
    "STOPWORDS_BASICAS",
    "resolver_caminho_arquivo",
    "carregar_csv",
    "carregar_dados",
    "obter_mapeamento_classes",
    "filtrar_rotulos_validos",
    "exibir_info_geral",
    "calcular_distribuicao_classes",
    "plotar_distribuicao_classes",
    "extrair_texto_bruto",
    "contar_tokens",
    "calcular_estatisticas_tamanho_textos",
    "calcular_estatisticas_tamanho_por_classe",
    "plotar_comprimento_textos",
    "calcular_palavras_frequentes_por_classe",
    "converter_para_dict_frequencias",
    "plotar_palavras_frequentes",
    "plotar_nuvem_palavras",
    "analisar_vocabulario",
    "analisar_duplicados", 
    "problemas_codificacao",
]
