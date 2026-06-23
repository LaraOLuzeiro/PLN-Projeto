# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Processamento de Linguagem Natural
# Prof. Tiago A. Almeida
#
#
# Nome:
# RA:
# ################################################################

"""
Modulo de Pre-processamento de Texto.

Fornece funcoes para:
  - Limpeza dos dados
  - POS Tagging
  - NER (Reconhecimento de Entidades Nomeadas)
  - Extracao de features POS/NER numericas para analise
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

import pandas as pd
import numpy as np

# ---------- dependencias opcionais ----------
try:
    import ftfy as _ftfy
    _FTFY_OK = True
except ImportError:
    _FTFY_OK = False

try:
    import nltk
    from nltk.corpus import stopwords as _nltk_sw
    try:
        _STOPWORDS_PT = set(_nltk_sw.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS_PT = set(_nltk_sw.words("portuguese"))
    _NLTK_OK = True
except ImportError:
    _STOPWORDS_PT = set()
    _NLTK_OK = False

try:
    import spacy as _spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False
# --------------------------------------------

# Stopwords adicionais especificas do dominio juridico
_STOPWORDS_JURIDICAS = {
    "senhor", "senhora", "excelentissimo", "vossa", "excelencia",
    "meritissimo", "ilustrissimo", "digno", "nobre",
    "ante", "apos", "sobre", "sob", "perante", "versus",
    "inc", "par", "art", "pagina", "documento",
}

# Conjunto unificado de stopwords (NLTK PT + juridicas)
STOPWORDS = _STOPWORDS_PT | _STOPWORDS_JURIDICAS

# Padroes de referencias legais para normalizacao
PADROES_LEGAIS = [
    (r"\b(?:ARTIGO_|artigo\s+)(\d+[A-Z0-9]*)\b", r"artigo\1"),
    (r"\b(?:LEI_|lei\s+)(\d+[A-Z0-9]*)\b", r"lei\1"),
    (r"\b(?:DECRETO_|decreto\s+)(\d+[A-Z0-9]*)\b", r"decreto\1"),
    (r"\b(?:EMENDA_|emenda\s+)(\d+[A-Z0-9]*)\b", r"emenda\1"),
]


# ==============================================================
# EXTRACAO DO CAMPO BODY
# ==============================================================

def extrair_texto_json(texto: str) -> str:
    """
    Remove o wrapper JSON do campo Body, se presente.
    Formato no CSV: {"conteudo do texto"} -> 'conteudo do texto'.
    """
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    if texto.startswith("{") and texto.endswith("}"):
        texto = texto[1:-1]
    texto = texto.strip('"')
    return texto


# ==============================================================
# CORRECAO DE ENCODING
# ==============================================================

def corrigir_encoding(texto: str) -> str:
    """
    Corrige problemas de double-encoding UTF-8/Latin-1 tipicos de PDFs/OCR.
    Usa ftfy quando disponivel (mais robusto); caso contrario aplica fallback manual.
    Exemplo: 'ÃƒÂ£' -> 'a~' (caractere com til).
    """
    if _FTFY_OK:
        return _ftfy.fix_text(str(texto))
    # Fallback: tenta corrigir sequencia mal codificada latin1->utf-8
    texto = str(texto)
    marcadores_mojibake = ("Ã", "Â", "\xc2", "\xc3")
    if not any(m in texto for m in marcadores_mojibake):
        return texto
    try:
        candidato = texto.encode("latin1").decode("utf-8")
        if candidato.count("Ã") < texto.count("Ã"):
            return candidato
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return texto


def normalizar_unicode(texto: str) -> str:
    """
    Normaliza o texto para forma NFC, garantindo consistencia de representacao
    de caracteres acentuados (ex.: 'a' + acento separado -> 'a com acento').
    """
    return unicodedata.normalize("NFC", str(texto))


def remover_acentos(texto: str) -> str:
    """Remove acentos do texto via decomposicao NFKD."""
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto_normalizado if not unicodedata.combining(c)
    )


# ==============================================================
# LIMPEZA DE RUIDO OCR
# ==============================================================

def limpar_ruido_ocr(texto: str) -> str:
    """
    Remove artefatos tipicos da extracao OCR de PDFs juridicos:
    - Wrapper JSON do campo Body
    - Problemas de encoding
    - Normalizacao Unicode
    - Sequencias de separadores repetidos
    - Numeros de pagina
    - Emails e URLs
    - Numeros de processo (sequencias longas de digitos)
    - Referencias legais ja normalizadas (ARTIGO_NN, LEI_NN, etc.)
    - Caracteres especiais isolados
    """
    texto_limpo = extrair_texto_json(str(texto))
    texto_limpo = corrigir_encoding(texto_limpo)
    texto_limpo = normalizar_unicode(texto_limpo)

    # Sequencias de separadores repetidos (ex.: "---", "===")
    texto_limpo = re.sub(r"[_\-=]{3,}", " ", texto_limpo)

    # Numeros de pagina (ex.: "pagina 3 de 10")
    texto_limpo = re.sub(
        r"\bpagina?\s*\d+\s*(de\s*\d+)?\b", " ", texto_limpo, flags=re.IGNORECASE
    )

    # Emails
    texto_limpo = re.sub(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", " EMAIL ", texto_limpo
    )

    # URLs
    texto_limpo = re.sub(r"https?://\S+|www\.\S+", " URL ", texto_limpo)

    # Numeros de processo (sequencias longas de digitos)
    texto_limpo = re.sub(r"\b\d{7,}\b", " NUM_PROC ", texto_limpo)

    # Normaliza marcadores de artigos juridicos
    for padrao, substituicao in PADROES_LEGAIS:
        texto_limpo = re.sub(padrao, substituicao, texto_limpo, flags=re.IGNORECASE)

    # Identifica CPF/CNPJ/CEP como tokens genericos
    texto_limpo = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", " CPF ", texto_limpo)
    texto_limpo = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", " CNPJ ", texto_limpo)

    # Remove separadores e caracteres de controle
    texto_limpo = re.sub(r"[_/\\|]+", " ", texto_limpo)
    texto_limpo = re.sub(r"[^\w\s]", " ", texto_limpo)
    texto_limpo = re.sub(r"\s+", " ", texto_limpo)
    return texto_limpo.strip()


# ==============================================================
# STOPWORDS
# ==============================================================

@lru_cache(maxsize=1)
def obter_stopwords_portugues() -> frozenset:
    """
    Retorna o conjunto unificado de stopwords:
    NLTK portugues + termos juridicos adicionais (cacheado).
    """
    return frozenset(STOPWORDS)


# ==============================================================
# MODELO SPACY
# ==============================================================

@lru_cache(maxsize=1)
def carregar_modelo_spacy(nome_modelo: str = "pt_core_news_lg"):
    """
    Carrega o modelo spaCy (cacheado).
    Desabilita 'parser' para maior velocidade.
    """
    if not _SPACY_OK:
        raise ImportError(
            "spaCy nao instalado. Execute: pip install spacy && "
            "python -m spacy download pt_core_news_lg"
        )
    nlp = _spacy.load(nome_modelo, disable=["parser"])
    nlp.max_length = 2_000_000

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, after="ner")
        padroes = [
            # Forçando siglas jurídicas comuns como ORG 
            {"label": "ORG", "pattern": [{"lower": "stf"}]},
            {"label": "ORG", "pattern": [{"lower": "stj"}]},
            {"label": "ORG", "pattern": [{"lower": "tst"}]},
            {"label": "ORG", "pattern": [{"lower": "tse"}]},
            {"label": "ORG", "pattern": [{"lower": "mp"}]},
            {"label": "ORG", "pattern": [{"lower": "mpf"}]},
            {"label": "ORG", "pattern": [{"lower": "mpe"}]},
            {"label": "ORG", "pattern": [{"lower": "inss"}]},
            {"label": "ORG", "pattern": [{"lower": "cemat"}]},
            {"label": "ORG", "pattern": [{"lower": "união"}, {"lower": "federal"}]},
            {"label": "ORG", "pattern": [{"lower": "ministério"}, {"lower": "público"}]},
            {"label": "ORG", "pattern": [{"lower": "fazenda"}, {"lower": "nacional"}]},
            {"label": "ORG", "pattern": [{"lower": "caixa"}, {"lower": "econômica"}, {"lower": "federal"}]},
        ]
        ruler.add_patterns(padroes)

    return nlp


# ==============================================================
# TOKENIZACAO (laco for explicito conforme especificacao)
# ==============================================================

def tokenizar_texto(
    texto: str,
    converter_minusculas: bool = True,
    aplicar_remocao_acentos: bool = False,
) -> list[str]:
    """
    Tokeniza o texto em lista de tokens alfanumericos.
    Usa laco for para percorrer caracter a caracter (controle granular).
    Descarta tokens muito curtos (tamanho <= 1).
    """
    texto_processado = limpar_ruido_ocr(texto)

    if converter_minusculas:
        texto_processado = texto_processado.lower()

    if aplicar_remocao_acentos:
        texto_processado = remover_acentos(texto_processado)

    tokens = []
    token_atual = []

    for caractere in texto_processado:
        if caractere.isalnum():
            token_atual.append(caractere)
        else:
            if token_atual:
                tokens.append("".join(token_atual))
                token_atual = []

    if token_atual:
        tokens.append("".join(token_atual))

    return tokens


def remover_stopwords_tokens(
    tokens: Iterable[str],
    tamanho_minimo: int = 2,
    stopwords_personalizadas: Iterable[str] | None = None,
) -> list[str]:
    """
    Remove stopwords de uma lista de tokens usando laco for.
    Controle granular: filtra por tamanho minimo, tokens numericos e stopwords.
    """
    conjunto_sw = set(obter_stopwords_portugues())

    if stopwords_personalizadas is not None:
        for termo in stopwords_personalizadas:
            conjunto_sw.add(termo)

    tokens_filtrados = []
    for token in tokens:
        if len(token) < tamanho_minimo:
            continue
        if token.isnumeric():
            continue
        if token in conjunto_sw:
            continue
        tokens_filtrados.append(token)

    return tokens_filtrados


# ==============================================================
# LEMATIZACAO (spaCy)
# ==============================================================

def lematizar_tokens(
    tokens: Iterable[str],
    nlp=None,
    nome_modelo: str = "pt_core_news_lg",
) -> list[str]:
    """
    Lematiza lista de tokens com spaCy usando laco for.
    Reduz cada token a sua forma canonica (ex.: 'recursos' -> 'recurso').
    Retorna lista de lemas em minusculas.
    """
    if nlp is None:
        nlp = carregar_modelo_spacy(nome_modelo=nome_modelo)

    texto = " ".join(tokens)
    if not texto.strip():
        return []

    doc = nlp(texto[:15000])  # limite por performance

    lemas = []
    for token in doc:
        if token.is_space:
            continue
        lema = token.lemma_.strip().lower()
        if lema and lema != "-pron-":
            lemas.append(lema)

    return lemas


# ==============================================================
# PIPELINE COMPLETO
# ==============================================================

def preprocessar_texto(
    texto: str,
    nlp=None,
    aplicar_lematizacao: bool = False,
    aplicar_remocao_acentos: bool = False,
    stopwords_personalizadas: Iterable[str] | None = None,
) -> str:
    """
    Executa o pipeline completo de preprocessamento em um unico texto:
    1. Extrai conteudo do wrapper JSON
    2. Corrige encoding (ftfy ou fallback)
    3. Normaliza Unicode (NFC)
    4. Remove ruido OCR
    5. Normaliza (minusculas, caracteres especiais)
    6. Tokeniza (laco for)
    7. Remove stopwords (laco for)
    8. Lematiza com spaCy (opcional)
    9. Remove stopwords apos lematizacao (segunda passagem)
    Retorna string preprocessada.
    """
    tokens = tokenizar_texto(
        texto,
        converter_minusculas=True,
        aplicar_remocao_acentos=aplicar_remocao_acentos,
    )
    tokens = remover_stopwords_tokens(
        tokens, stopwords_personalizadas=stopwords_personalizadas
    )

    if aplicar_lematizacao and nlp is not None:
        tokens = lematizar_tokens(tokens, nlp=nlp)
        # Segunda passagem de remocao de stopwords apos lematizacao
        tokens = remover_stopwords_tokens(
            tokens, stopwords_personalizadas=stopwords_personalizadas
        )

    return " ".join(tokens)


def preprocessar_textos(
    textos: Iterable[str],
    nlp=None,
    aplicar_lematizacao: bool = False,
    aplicar_remocao_acentos: bool = False,
    stopwords_personalizadas: Iterable[str] | None = None,
) -> list[str]:
    """Aplica preprocessar_texto() em uma lista de textos."""
    textos_processados = []
    for texto in textos:
        textos_processados.append(
            preprocessar_texto(
                texto,
                nlp=nlp,
                aplicar_lematizacao=aplicar_lematizacao,
                aplicar_remocao_acentos=aplicar_remocao_acentos,
                stopwords_personalizadas=stopwords_personalizadas,
            )
        )
    return textos_processados


def aplicar_pipeline(
    df: pd.DataFrame,
    coluna: str = "Body",
    nlp=None,
    lematizar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aplica preprocessar_texto() em cada linha do DataFrame.
    Retorna DataFrame com coluna 'texto_preprocessado' adicionada.
    Exibe progresso a cada 2000 linhas quando verbose=True.
    """
    if verbose:
        modo = "com lematizacao (spaCy)" if lematizar and nlp is not None else "sem lematizacao"
        print(f"Preprocessando {len(df):,} textos ({modo})...")

    textos_proc = []
    for i, texto in enumerate(df[coluna]):
        textos_proc.append(
            preprocessar_texto(texto, nlp=nlp, aplicar_lematizacao=lematizar)
        )
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1:>6,} / {len(df):,} processados")

    df_result = df.copy()
    # np.array garante tipo numpy puro (evita PyArrow backend incompativel com sklearn)
    df_result["texto_preprocessado"] = np.array(textos_proc, dtype=object)

    if verbose:
        print("Preprocessamento concluido.")
    return df_result


# ==============================================================
# REMOCAO DE DUPLICATAS E TEXTOS VAZIOS
# ==============================================================

def remover_documentos_duplicados(df: pd.DataFrame, coluna: str = "Body") -> pd.DataFrame:
    """
    Remove documentos duplicados baseados na coluna de texto bruto para evitar data leakage.
    """
    df_limpo = df.drop_duplicates(subset=[coluna]).copy()
    return df_limpo

def remover_textos_vazios(df: pd.DataFrame, coluna: str = "texto_preprocessado") -> pd.DataFrame:
    """
    Remove documentos que ficaram vazios ou compostos apenas de espaços após o pré-processamento.
    """
    # Filtra mantendo apenas linhas onde a string não é vazia após o strip()
    df_limpo = df[df[coluna].str.strip().astype(bool)].copy()
    return df_limpo


# ==============================================================
# VERIFICACAO DE RUIDOS APOS LIMPEZA
# ==============================================================

def verificar_ruidos_dataset(df: pd.DataFrame, coluna_texto: str = 'texto_preprocessado', coluna_alvo: str = 'Category'):
    """
    Realiza uma varredura no dataset para identificar possíveis ruídos residuais 
    que possam quebrar o treinamento dos modelos.
    """
    print("=== VERIFICAÇÃO DE RUÍDOS NO DATASET PÓS-PROCESSAMENTO ===\n")
    
    # 1. Valores Nulos (NaN)
    nulos = df[coluna_texto].isna().sum()
    print(f"1. Documentos nulos (NaN): {nulos}")
    
    # 2. Textos Vazios (apenas espaços ou string vazia)
    # Convertendo para string antes para evitar erros com possíveis NaNs
    vazios = (df[coluna_texto].astype(str).str.strip() == "").sum()
    print(f"2. Documentos totalmente vazios (''): {vazios}")
    
    # 3. Rótulos Inválidos (Verifica se sobrou algum -1)
    rotulos_invalidos = (df[coluna_alvo] < 0).sum()
    print(f"3. Rótulos inválidos (Category < 0): {rotulos_invalidos}")
    
    # 4. Duplicatas no texto processado
    duplicatas_proc = df.duplicated(subset=[coluna_texto]).sum()
    print(f"4. Duplicatas no texto processado: {duplicatas_proc}")
        
    # 5. Estatísticas de Tokens (Identificar textos curtíssimos, ex: sobrou só 1 palavra)
    contagem_tokens = df[coluna_texto].astype(str).apply(lambda x: len(x.split()))
    min_tokens = contagem_tokens.min()
    max_tokens = contagem_tokens.max()
    textos_muito_curtos = (contagem_tokens < 3).sum() # Define < 3 palavras como muito curto
    
    print(f"\n5. Estatísticas de comprimento (tokens):")
    print(f"   - Mínimo de tokens em um documento: {min_tokens}")
    print(f"   - Máximo de tokens em um documento: {max_tokens}")
    print(f"   - Documentos com menos de 3 tokens úteis: {textos_muito_curtos}")
    
    # Conclusão automatizada
    print("\n" + "="*58)
    if nulos == 0 and vazios == 0 and rotulos_invalidos == 0 and textos_muito_curtos == 0:
        print("Dataset limpo e estruturado.")
    else:
        print("ATENÇÃO: Foram encontrados ruídos no dataset.")


# ==============================================================
# POS TAGGING
# ==============================================================

def aplicar_pos_tagging(
    texto: str,
    nlp=None,
    nome_modelo: str = "pt_core_news_lg",
) -> list[tuple[str, str, str]]:
    """
    Aplica Part-of-Speech tagging com spaCy em um texto.
    Retorna lista de tuplas (token, pos_tag_universal, pos_tag_fino).
    Ex.: ('recurso', 'NOUN', 'NN')
    """
    if nlp is None:
        nlp = carregar_modelo_spacy(nome_modelo=nome_modelo)
    doc = nlp(str(texto)[:10000])
    pares = []
    for token in doc:
        if not token.is_space and not token.is_punct:
            pares.append((token.text, token.pos_, token.tag_))
    return pares


# ==============================================================
# NER - RECONHECIMENTO DE ENTIDADES NOMEADAS
# ==============================================================

def extrair_entidades_nomeadas(
    texto: str,
    nlp=None,
    nome_modelo: str = "pt_core_news_lg", 
) -> list[dict]:
    """
    Extrai entidades nomeadas (NER) usando o modelo pt_core_news_lg do spaCy.
    O modelo pode identificar diferentes categorias, como pessoas (PER),
    organizações (ORG), localidades (LOC) e outras classes. 
    """
    if nlp is None:
        nlp = carregar_modelo_spacy(nome_modelo=nome_modelo)
    doc = nlp(str(texto)[:10000])
    entidades = []
    for ent in doc.ents:
        texto_ent = ent.text.strip().lower()
        
        # Ignorar artigos/leis que foram capturados como MISC
        if re.match(r"^(artigo|art|lei|decreto|emenda|sumula)\s*\d*", texto_ent):
            continue
            
        # Ignorar entidades "MISC" muito longas
        if ent.label_ == "MISC" and len(texto_ent.split()) > 3:
            continue

        # Ignorar se a entidade for uma stopword solta
        if texto_ent in STOPWORDS:
            continue

        entidades.append({
            "texto": ent.text,
            "tipo": ent.label_,
            "inicio": ent.start_char,
            "fim": ent.end_char,
        })
    return entidades


def extrair_mencoes_legais(
    texto: str,
    nlp=None,
    nome_modelo: str = "pt_core_news_lg", 
) -> dict[str, list[str]]:
    """
    Extrai mencoes a entidades juridicas e referencias legais do texto.
    Combina NER do spaCy (entidades formais) com regex para mencoes legais
    (artigos, leis, decretos) que frequentemente nao sao capturados pelo NER.
    """
    padrao_legal = re.compile(
        r"\b(?:artigo|art|lei|decreto|sumula|constituicao|emenda)\s*\d+[a-z0-9\u00BA]*",
        flags=re.IGNORECASE,
    )

    entidades_spacy = []
    if nlp is not None:
        doc = nlp(str(texto)[:10000])
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PER", "LOC", "MISC"}:
                entidades_spacy.append(ent.text)

    mencoes_regex = padrao_legal.findall(str(texto))
    return {
        "entidades_spacy": sorted(set(entidades_spacy)),
        "mencoes_legais_regex": sorted(set(mencoes_regex)),
    }


# ==============================================================
# FEATURES PLN (para analise e engenharia de atributos)
# ==============================================================

def extrair_features_pos(
    df: pd.DataFrame,
    nlp,
    coluna: str = "texto_preprocessado",
    amostra: int = 3000,
) -> pd.DataFrame:
    """
    Extrai proporcoes de tags POS como features numericas por documento.
    Util para analise das diferencas gramaticais entre classes e para
    engenharia de atributos em modelos classicos.
    Colunas resultantes: Category, prop_NOUN, prop_VERB, prop_ADJ, prop_PROPN, prop_NUM, prop_ADV.
    """
    df_amostra = df[df["Category"] >= 0].sample(
        min(amostra, len(df)), random_state=42
    )

    registros = []
    for _, row in df_amostra.iterrows():
        pares = aplicar_pos_tagging(str(row[coluna])[:5000], nlp=nlp)
        contagem: dict[str, int] = {}
        for _, pos, _ in pares:
            contagem[pos] = contagem.get(pos, 0) + 1
        total = max(len(pares), 1)
        registros.append({
            "Category": row["Category"],
            "prop_NOUN": contagem.get("NOUN", 0) / total,
            "prop_VERB": contagem.get("VERB", 0) / total,
            "prop_ADJ": contagem.get("ADJ", 0) / total,
            "prop_PROPN": contagem.get("PROPN", 0) / total,
            "prop_NUM": contagem.get("NUM", 0) / total,
            "prop_ADV": contagem.get("ADV", 0) / total,
        })

    return pd.DataFrame(registros)


def extrair_features_ner(
    df: pd.DataFrame,
    nlp,
    coluna: str = "texto_preprocessado",
    amostra: int = 1000,
) -> pd.DataFrame:
    """
    Extrai contagens e proporcoes de tipos de entidades NER como features.
    Entidades tipicas no dominio juridico:
    - ORG: organizacoes
    - PER: pessoas
    - LOC: localidades
    - MISC: entidades diversas
    """
    df_amostra = df[df["Category"] >= 0].sample(
        min(amostra, len(df)), random_state=42
    )

    registros = []
    for _, row in df_amostra.iterrows():
        ents = extrair_entidades_nomeadas(str(row[coluna])[:5000], nlp=nlp)
        contagem: dict[str, int] = {}
        for ent in ents:
            contagem[ent["tipo"]] = contagem.get(ent["tipo"], 0) + 1
            
        total = max(len(ents), 1)
        registros.append({
            "Category": row["Category"],
            "n_entidades": len(ents),
            "prop_PER": contagem.get("PER", 0) / total,
            "prop_ORG": contagem.get("ORG", 0) / total,
            "prop_LOC": contagem.get("LOC", 0) / total,
            "prop_MISC": contagem.get("MISC", 0) / total,
        })

    return pd.DataFrame(registros)


def plotar_distribuicao_pos(
    df_pos: pd.DataFrame,
    salvar_em: str | None = "figs/distribuicao_pos.png",
) -> None:
    """
    Plota proporcoes medias de tags POS por classe em grafico de barras agrupadas.
    """
    import os
    import matplotlib.pyplot as plt
    os.makedirs("figs", exist_ok=True)

    nomes_classes = {0: "Acordao", 1: "ARE", 2: "Despacho", 3: "RE", 4: "Sentenca"}
    colunas_pos = [c for c in df_pos.columns if c.startswith("prop_")]
    df_pos = df_pos.copy()
    df_pos["nome_classe"] = df_pos["Category"].map(nomes_classes)

    media_pos = df_pos.groupby("nome_classe")[colunas_pos].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    media_pos.T.plot(kind="bar", ax=ax, colormap="Set2", width=0.7, alpha=0.9)
    ax.set_title("Proporcao de Tags POS por Classe (media)", fontweight="bold")
    ax.set_xlabel("Tag POS")
    ax.set_ylabel("Proporcao Media")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


__all__ = [
    "STOPWORDS",
    "extrair_texto_json",
    "corrigir_encoding",
    "normalizar_unicode",
    "remover_acentos",
    "limpar_ruido_ocr",
    "obter_stopwords_portugues",
    "carregar_modelo_spacy",
    "tokenizar_texto",
    "remover_stopwords_tokens",
    "lematizar_tokens",
    "preprocessar_textos",
    "aplicar_pipeline",
    "aplicar_pos_tagging",
    "extrair_entidades_nomeadas",
    "extrair_mencoes_legais",
    "extrair_features_pos",
    "extrair_features_ner",
    "plotar_distribuicao_pos",
    "remover_documentos_duplicados",
    "remover_textos_vazios",
    "verificar_ruidos_dataset"
]
