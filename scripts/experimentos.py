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

"""
Modulo de Experimentos e Modelos.

Contem funcoes para:
  - Divisao treino/validacao estratificada
  - Vetorizacao TF-IDF (esparsa) e Word2Vec (densa)
  - Modelos classicos: Regressao Logistica, SVM Linear, Naive Bayes
  - Validacao cruzada estratificada (5-fold) sem data leakage
  - PLN basico: NER juridico com spaCy
  - Modelo profundo: BiLSTM com mecanismo de atencao (2 camadas, PyTorch)
  - Transformer: fine-tuning do BERTimbau (HuggingFace)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

# ---------- dependencias opcionais ----------
try:
    from gensim.models import Word2Vec as _Word2Vec
    _GENSIM_OK = True
    _GENSIM_IMPORT_ERROR = None
except Exception as exc:
    _Word2Vec = None
    _GENSIM_OK = False
    _GENSIM_IMPORT_ERROR = exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        get_linear_schedule_with_warmup,
    )
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

try:
    import spacy as _spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False
# --------------------------------------------


MAPEAMENTO_CLASSES = {
    0: "Acordao",
    1: "ARE",
    2: "Despacho",
    3: "RE",
    4: "Sentenca",
}


# ==============================================================
# DIVISAO DOS DADOS
# ==============================================================

def dividir_treino_validacao(
    textos: Iterable[str],
    rotulos: Iterable[int],
    proporcao_validacao: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """
    Divide dados em treino/validacao de forma estratificada.
    Estratificacao garante que a proporcao de cada classe seja mantida
    em ambas as particoes, essencial para datasets desbalanceados.
    """
    return train_test_split(
        list(textos),
        np.asarray(list(rotulos)),
        test_size=proporcao_validacao,
        random_state=random_state,
        stratify=np.asarray(list(rotulos)),
    )


# ==============================================================
# REPRESENTACAO ESPARSA - TF-IDF
# ==============================================================

def criar_vetorizador_tfidf(
    max_features: int = 50000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95, # ignora termos presentes em quase todos os documentos (ruido)
) -> TfidfVectorizer:
    """
    Cria vetorizador TF-IDF com configuracoes otimizadas para textos juridicos.
    - sublinear_tf: aplica log(1+TF) para suavizar palavras muito frequentes
    - ngram_range (1,2): unigrams + bigrams capturam frases juridicas-chave
    - strip_accents: normaliza caracteres acentuados
    - min_df=2: ignora termos que apareceram em apenas 1 documento (ruido)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-zA-Z\u00C0-\u00FF_]{2,}\b",
    )


def vetorizar_tfidf(
    textos_treino: Iterable[str],
    textos_validacao: Iterable[str] | None = None,
    textos_teste: Iterable[str] | None = None,
    **parametros_tfidf,
):
    """
    Ajusta o TF-IDF no treino e transforma as demais particoes.
    Retorna: (vetorizador, X_treino, [X_val], [X_teste]).
    """
    vetorizador = criar_vetorizador_tfidf(**parametros_tfidf)
    X_treino = vetorizador.fit_transform(list(textos_treino))

    resultados = [vetorizador, X_treino]

    if textos_validacao is not None:
        resultados.append(vetorizador.transform(list(textos_validacao)))
    if textos_teste is not None:
        resultados.append(vetorizador.transform(list(textos_teste)))

    print(f"  TF-IDF | vocab: {len(vetorizador.vocabulary_):,} | features: {X_treino.shape[1]:,}")
    return tuple(resultados)


# ==============================================================
# REPRESENTACAO DENSA - WORD2VEC
# ==============================================================

def tokenizar_para_word2vec(textos: Iterable[str]) -> list[list[str]]:
    """Converte lista de textos preprocessados em lista de listas de tokens."""
    sentencas = []
    for texto in textos:
        tokens = str(texto).split()
        if tokens:
            sentencas.append(tokens)
    return sentencas


def treinar_word2vec(
    textos: Iterable[str],
    dimensao: int = 100,
    janela: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epocas: int = 10,
    seed: int = 42,
    sg: int = 0,
):
    """
    Treina modelo Word2Vec no corpus juridico.
    - sg=0: CBOW (mais rapido e adequado para corpus grande)
    - sg=1: Skip-gram (melhor para vocabulario raro)
    Retorna modelo Word2Vec treinado.
    """
    if not _GENSIM_OK:
        raise ImportError(
            "Nao foi possivel importar gensim/Word2Vec. "
            "Execute a celula de configuracao do ambiente para instalar "
            "gensim==4.3.3 e scipy==1.11.4, reinicie o kernel e rode os imports novamente. "
            f"Erro original: {_GENSIM_IMPORT_ERROR}"
        ) from _GENSIM_IMPORT_ERROR

    sentencas = tokenizar_para_word2vec(textos)
    modelo = _Word2Vec(
        sentences=sentencas,
        vector_size=dimensao,
        window=janela,
        min_count=min_count,
        workers=workers,
        epochs=epocas,
        sg=sg,
        seed=seed,
    )
    print(f"  Word2Vec | vocab: {len(modelo.wv):,} | dim: {dimensao}")
    return modelo


def vetorizar_word2vec_media(
    textos: Iterable[str],
    modelo_word2vec,
) -> np.ndarray:
    """
    Representa cada documento como a media dos vetores Word2Vec de seus tokens
    (Bag-of-Embeddings). Tokens ausentes do vocabulario sao ignorados.
    Usa laco for para calcular a media elemento a elemento.
    """
    dimensao = modelo_word2vec.vector_size
    vetores = []

    for texto in textos:
        tokens = str(texto).split()
        vetores_validos = []
        for token in tokens:
            if token in modelo_word2vec.wv:
                vetores_validos.append(modelo_word2vec.wv[token])

        if vetores_validos:
            vetor_medio = np.zeros(dimensao)
            for v in vetores_validos:
                vetor_medio = vetor_medio + v
            vetor_medio = vetor_medio / len(vetores_validos)
        else:
            vetor_medio = np.zeros(dimensao, dtype=float)

        vetores.append(vetor_medio)

    return np.asarray(vetores)


# ==============================================================
# MODELOS CLASSICOS
# ==============================================================

def treinar_regressao_logistica(
    X_treino,
    y_treino,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Regressao Logistica Multinomial com pesos de classe balanceados.
    class_weight='balanced': compensa o forte desbalanceamento (RE ~56%).
    solver='lbfgs': adequado para datasets medios com multiclasse.
    """
    modelo = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    modelo.fit(X_treino, y_treino)
    return modelo


def treinar_svm_linear(
    X_treino,
    y_treino,
    C: float = 1.0,
    calibrar: bool = True,
    random_state: int = 42,
):
    """
    SVM Linear com pesos balanceados.
    Se calibrar=True, usa CalibratedClassifierCV (Platt scaling) para
    obter estimativas de probabilidade via predict_proba().
    LinearSVC e mais eficiente para dados TF-IDF esparsos de alta dimensao.
    """
    svm_base = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    if calibrar:
        modelo = CalibratedClassifierCV(svm_base, cv=3, n_jobs=-1)
    else:
        modelo = svm_base
    modelo.fit(X_treino, y_treino)
    return modelo


def treinar_naive_bayes(
    X_treino,
    y_treino,
    alpha: float = 1.0,
) -> MultinomialNB:
    """
    Naive Bayes Multinomial com suavizacao de Laplace (add-1 smoothing).
    O parametro alpha=1.0 garante que nenhuma probabilidade seja zero
    para palavras nao vistas no treinamento (Laplace smoothing).
    Requer X com valores nao-negativos (TF-IDF sublinear_tf satisfaz).
    """
    modelo = MultinomialNB(alpha=alpha)
    modelo.fit(X_treino, y_treino)
    return modelo


def avaliar_modelo_em_validacao(
    modelo,
    X_validacao,
    y_validacao,
    nome_modelo: str,
    representacao: str,
) -> dict:
    """Avalia um modelo no conjunto de validacao e retorna metricas basicas."""
    predicoes = modelo.predict(X_validacao)
    return {
        "modelo": nome_modelo,
        "representacao": representacao,
        "accuracy": float(accuracy_score(y_validacao, predicoes)),
        "f1_macro": float(f1_score(y_validacao, predicoes, average="macro")),
        "f1_weighted": float(f1_score(y_validacao, predicoes, average="weighted")),
    }


def avaliar_validacao_cruzada(
    X_texto: list,
    y: np.ndarray,
    modelos_dict: dict,
    n_splits: int = 5,
    max_features: int = 30000,
) -> dict:
    """
    Avaliacao com StratifiedKFold para manter proporcao de classes em cada fold.
    A vetorizacao TF-IDF e re-feita dentro de cada fold para evitar data leakage:
    o vocabulario e ajustado apenas nos dados de treino do fold.
    Retorna dicionario com F1-Macro medio e desvio por modelo.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados: dict = {}
    X_array = np.array(X_texto)
    y_array = np.array(y)

    for nome, modelo in modelos_dict.items():
        scores = []
        for idx_tr, idx_vl in skf.split(X_array, y_array):
            X_tr = X_array[idx_tr].tolist()
            X_vl = X_array[idx_vl].tolist()
            y_tr = y_array[idx_tr]
            y_vl = y_array[idx_vl]

            vet = TfidfVectorizer(
                max_features=max_features, ngram_range=(1, 2),
                sublinear_tf=True, min_df=2,
            )
            X_tr_vet = vet.fit_transform(X_tr)
            X_vl_vet = vet.transform(X_vl)

            modelo.fit(X_tr_vet, y_tr)
            preds = modelo.predict(X_vl_vet)
            scores.append(f1_score(y_vl, preds, average="macro"))

        media = float(np.mean(scores))
        desvio = float(np.std(scores))
        resultados[nome] = {"scores": scores, "media": media, "desvio": desvio}
        print(f"  {nome:<35}: F1-Macro = {media:.4f} +/- {desvio:.4f}")

    return resultados


@dataclass
class ResultadoExperimento:
    """Armazena modelo, representacao e metricas de um experimento."""
    nome_modelo: str
    representacao: str
    artefato_modelo: object
    artefato_representacao: object
    metricas: dict


def executar_baselines_classicos(
    textos_treino: Iterable[str],
    y_treino,
    textos_validacao: Iterable[str],
    y_validacao,
) -> tuple[pd.DataFrame, dict[str, ResultadoExperimento]]:
    """
    Treina e avalia todos os modelos classicos (LR, SVM, NB) com TF-IDF
    e LR com Word2Vec. Retorna tabela de resultados e artefatos.
    """
    resultados = []
    artefatos: dict[str, ResultadoExperimento] = {}

    # --- TF-IDF ---
    vet, X_tr_tfidf, X_vl_tfidf = vetorizar_tfidf(textos_treino, textos_validacao)

    # Naive Bayes requer valores nao-negativos (MaxAbsScaler mantem esparsidade)
    scaler = MaxAbsScaler()
    X_tr_nn = scaler.fit_transform(X_tr_tfidf)
    X_vl_nn = scaler.transform(X_vl_tfidf)

    modelos_tfidf = {
        "Regressao Logistica (TF-IDF)": (
            treinar_regressao_logistica(X_tr_tfidf, y_treino), vet
        ),
        "SVM Linear (TF-IDF)": (
            treinar_svm_linear(X_tr_tfidf, y_treino), vet
        ),
        "Naive Bayes (TF-IDF)": (
            treinar_naive_bayes(X_tr_nn, y_treino, alpha=1.0),
            (scaler, vet),
        ),
    }

    for nome_modelo, (modelo, artefato_rep) in modelos_tfidf.items():
        if nome_modelo == "Naive Bayes (TF-IDF)":
            X_vl = X_vl_nn
        else:
            X_vl = X_vl_tfidf
        metricas = avaliar_modelo_em_validacao(
            modelo, X_vl, y_validacao,
            nome_modelo=nome_modelo, representacao="TF-IDF",
        )
        artefatos[nome_modelo] = ResultadoExperimento(
            nome_modelo=nome_modelo,
            representacao="TF-IDF",
            artefato_modelo=modelo,
            artefato_representacao=artefato_rep,
            metricas=metricas,
        )
        resultados.append(metricas)

    # --- Word2Vec ---
    if _GENSIM_OK:
        modelo_w2v = treinar_word2vec(textos_treino)
        X_tr_w2v = vetorizar_word2vec_media(textos_treino, modelo_w2v)
        X_vl_w2v = vetorizar_word2vec_media(textos_validacao, modelo_w2v)

        modelo_lr_w2v = treinar_regressao_logistica(X_tr_w2v, y_treino)
        metricas_w2v = avaliar_modelo_em_validacao(
            modelo_lr_w2v, X_vl_w2v, y_validacao,
            nome_modelo="LR + Word2Vec", representacao="Word2Vec",
        )
        artefatos["LR + Word2Vec"] = ResultadoExperimento(
            nome_modelo="LR + Word2Vec",
            representacao="Word2Vec",
            artefato_modelo=modelo_lr_w2v,
            artefato_representacao=modelo_w2v,
            metricas=metricas_w2v,
        )
        resultados.append(metricas_w2v)

    return pd.DataFrame(resultados).sort_values("f1_macro", ascending=False), artefatos



# ==============================================================
# PLN BASICO - NER JURIDICO
# ==============================================================

PADRAO_MENCOES_LEGAIS = re.compile(
    r"\b(?:artigo|art|lei|decreto|sumula|constituicao|emenda)\s+\d+[a-z0-9\u00BA]*",
    flags=re.IGNORECASE,
)


def aplicar_ner_juridico(
    textos: Iterable[str],
    nlp,
    n_amostras: int = 500,
) -> tuple[Counter, dict]:
    """
    Aplica NER (spaCy) em amostra dos textos e reporta entidades encontradas.
    Identifica mencoes a tribunais (ORG), partes (PER) e localidades (LOC).
    Retorna contagem por tipo e exemplos por tipo.
    """
    amostra = list(textos)[:n_amostras]
    contagem_tipos: Counter = Counter()
    exemplos_por_tipo: dict[str, list] = {}

    for texto in amostra:
        doc = nlp(str(texto)[:5000])
        for ent in doc.ents:
            contagem_tipos[ent.label_] += 1
            if ent.label_ not in exemplos_por_tipo:
                exemplos_por_tipo[ent.label_] = []
            if len(exemplos_por_tipo[ent.label_]) < 4:
                exemplos_por_tipo[ent.label_].append(ent.text)

    print(f"\nNER - top entidades em {n_amostras} amostras:")
    for tipo, n in contagem_tipos.most_common(10):
        exs = exemplos_por_tipo.get(tipo, [])
        print(f"  {tipo:<10}: {n:>5}  | ex: {', '.join(exs[:3])}")

    return contagem_tipos, exemplos_por_tipo


def extrair_mencoes_legais_amostra(
    texto: str,
    nlp=None,
) -> dict[str, list[str]]:
    """
    Extrai entidades (spaCy) e mencoes legais (regex) de um texto.
    Interface simplificada para uso no notebook.
    """
    entidades = []
    if nlp is not None and _SPACY_OK:
        doc = nlp(str(texto)[:10000])
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PER", "LOC", "MISC"}:
                entidades.append(ent.text)

    mencoes_regex = PADRAO_MENCOES_LEGAIS.findall(str(texto))
    return {
        "entidades_spacy": sorted(set(entidades)),
        "mencoes_legais_regex": sorted(set(mencoes_regex)),
    }


# ==========================  FUNÇÕES PARA BiLSTM e BERT ==========================

def construir_vocabulario(
    textos: Iterable[str],
    max_vocabulario: int = 30000,
    minimo_frequencia: int = 2,
) -> dict[str, int]:
    """
    Constroi vocabulario a partir de textos ja preprocessados (tokens separados por espaco).
    Retorna dicionario {token: id} com tokens <pad>=0 e <unk>=1.
    """
    frequencias: dict[str, int] = {}
    for texto in textos:
        for token in str(texto).split():
            frequencias[token] = frequencias.get(token, 0) + 1

    vocabulario_ordenado = sorted(
        [(t, f) for t, f in frequencias.items() if f >= minimo_frequencia],
        key=lambda x: x[1],
        reverse=True,
    )

    vocabulario: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for indice, (token, _) in enumerate(vocabulario_ordenado[: max_vocabulario - 2], start=2):
        vocabulario[token] = indice

    return vocabulario


def codificar_texto(
    texto: str,
    vocabulario: dict[str, int],
    comprimento_maximo: int = 300,
) -> list[int]:
    """Codifica texto em lista de IDs do vocabulario com padding/truncamento."""
    tokens = str(texto).split()
    sequencia = []
    for token in tokens[:comprimento_maximo]:
        sequencia.append(vocabulario.get(token, vocabulario.get("<unk>", 1)))
    while len(sequencia) < comprimento_maximo:
        sequencia.append(0)
    return sequencia


if _TORCH_OK:

    class DatasetSequencias(Dataset):
        """Dataset PyTorch para sequencias de texto com ou sem rotulos."""
        def __init__(
            self,
            textos: Iterable[str],
            rotulos: Iterable[int] | None,
            vocabulario: dict[str, int],
            comprimento_maximo: int = 300,
        ) -> None:
            self.textos = list(textos)
            self.rotulos = None if rotulos is None else list(rotulos)
            self.vocabulario = vocabulario
            self.comprimento_maximo = comprimento_maximo

        def __len__(self) -> int:
            return len(self.textos)

        def __getitem__(self, indice: int):
            entrada = codificar_texto(
                self.textos[indice], self.vocabulario, self.comprimento_maximo
            )
            item = {"input_ids": torch.tensor(entrada, dtype=torch.long)}
            if self.rotulos is not None:
                item["labels"] = torch.tensor(self.rotulos[indice], dtype=torch.long)
            return item
else:
    # Stubs para quando PyTorch nao esta instalado
    class DatasetSequencias:  # type: ignore
        pass

try:
    import ftfy
    _FTFY_OK = True
except ImportError:
    _FTFY_OK = False
    print("Erro ao importar módulo ftfy")


def limpar_texto_bilstm_bert(texto: str) -> str:
    """
    Limpeza higienica minimalista focada em Redes Neurais.
    Não remove stopwords, não lematiza e mantém a pontução e acentos.
    Isso porque BiLSTM e BERT não precisam desses tipos de pre-processamento.
    """    

    # Tratamento de nulos e conversão para string
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()

    # Remocao do wrapper JSON ('{"conteudo"}')
    if texto.startswith("{") and texto.endswith("}"):
        texto = texto[1:-1]
    texto = texto.strip('"')

    # Correção de double-encoding (ex: 'ÃƒÂ£' -> 'a~')
    if _FTFY_OK:
        texto = ftfy.fix_text(texto)

    # Remoção de carimbos eletrônicos
    padrao_assinatura = re.compile(r"Este documento foi assinado digitalmente por[^\.]+\.", flags=re.IGNORECASE)
    texto = padrao_assinatura.sub(" ", texto)

    # Remoção de números de processos
    padrao_processo = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4}\b")
    texto = padrao_processo.sub(" NUM_PROCESSO ", texto)

    # Normalização de Espaços (transforma multiplos espacos/quebras em um so)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def fatiar_head_tail(texto: str, max_tokens: int = 500, tokens_head: int = 120) -> str:
    """
    Fatia o texto preservando o começo (relatório) e o final (decisão).
    Descarta infos (citaçoes de jurisprudencia) que confundem a rede.
    """
    # Quebra o texto em uma lista de palavras usando os espacos em branco
    tokens = texto.split()
    
    # Se o texto for menor que o limite, nao precisa cortar nada
    if len(tokens) <= max_tokens:
        return texto
        
    # Calcula quantos tokens vao ficar no final (Tail)
    tokens_tail = max_tokens - tokens_head
    
    # Fatiamento (Slicing) de listas no Python
    head = tokens[:tokens_head]
    tail = tokens[-tokens_tail:]
    
    # Cola as duas partes de volta, inserindo um marcador no meio
    tokens_fatiados = head + ["...[MIOLO_CORTADO]..."] + tail
    
    return " ".join(tokens_fatiados)



def preparar_dados_treino_bilstm_bert(df: pd.DataFrame, coluna_texto: str = "Body", coluna_alvo: str = "Category") -> pd.DataFrame:
    """
    Funcao orquestradora: remove classes invalidas e aplica o pipeline de DL.
    """

    qtd_missings_removidos = 0
    qtd_duplicatas_removidos = 0

    # Filtra a classe -1
    df_valido = df[df[coluna_alvo] != -1].copy()
    qtd_missings_removidos = len(df[df[coluna_alvo]==-1])

    # Remove duplicatas
    qtd_duplicatas_removidos = df_valido.duplicated(subset=[coluna_texto]).sum()
    df_valido = df_valido.drop_duplicates(subset=[coluna_texto], keep='first')

    # Reseta o indice do DataFrame apos o filtro
    df_valido = df_valido.reset_index(drop=True)
    
    # Aplica a limpeza higienica
    df_valido["texto_limpo"] = df_valido[coluna_texto].apply(limpar_texto_bilstm_bert)
    
    # Aplica o fatiamento Head + Tail
    df_valido["texto_limpo"] = df_valido["texto_limpo"].apply(lambda x: fatiar_head_tail(x, max_tokens=500, tokens_head=120))
    
    print(f"Missings (-1) removidos: {qtd_missings_removidos}")
    print(f"Duplicadas removidas pós remoção de missings: {qtd_duplicatas_removidos}")

    return df_valido


def preparar_dados_teste_bilstm_bert(df_teste: pd.DataFrame, coluna_texto: str = "Body") -> pd.DataFrame:
    """
    Função EXCLUSIVA PARA O TESTE.
    Aplica a mesma limpeza, mas não remove as duplicatas.
    """
    df_processado = df_teste.copy()
    
    # Aplica exatamente a mesma limpeza higienica do treino
    df_processado["texto_limpo"] = df_processado[coluna_texto].apply(limpar_texto_bilstm_bert)
    
    # Aplica exatamente o mesmo fatiamento Head + Tail do treino
    df_processado["texto_limpo"] = df_processado["texto_limpo"].apply(lambda x: fatiar_head_tail(x, max_tokens=500, tokens_head=120))
    
    return df_processado


def extrair_nao_rotulados_limpos(df_bruto: pd.DataFrame, coluna_texto: str = "Body", coluna_alvo: str = "Category") -> list[str]:
    """
    Filtra estritamente as peticoes sem rotulo (-1) do CSV original e aplica a limpeza básica.
    """
    df_nr = df_bruto[df_bruto[coluna_alvo] == -1].copy()
    return df_nr[coluna_texto].apply(limpar_texto_bilstm_bert).tolist()


def gerar_matriz_embedding_interna(
    textos_treino: list[str], 
    vocabulario: dict[str, int], 
    dimensao: int = 128,
    textos_extras_w2v: list[str] | None = None
) -> torch.Tensor:
    """
    Treina o Word2Vec somando o treino oficial com textos extras (ex: a classe -1) e monta a matriz PyTorch.
    """
    if not _GENSIM_OK:
        raise ImportError("Gensim nao instalado. Execute: pip install gensim")
    from gensim.models import Word2Vec as GensimWord2Vec

    # Junta a lista de textos do treino com a lista de textos -1
    textos_unificados = list(textos_treino)
    if textos_extras_w2v is not None:
        textos_unificados.extend(textos_extras_w2v)
        
    print(f"  [Word2Vec Interno] Treinando semantica com {len(textos_unificados):,} textos totais (Treino + Nao Rotulados)...")
    
    # Converte a lista de textos numa lista de listas de palavras
    sentencas = [str(t).split() for t in textos_unificados]
    
    modelo_w2v = GensimWord2Vec(
        sentences=sentencas, vector_size=dimensao, window=5, min_count=1, workers=4, epochs=15, seed=42
    )
    
    # Monta a matriz base do PyTorch com pequenos valores aleatorios
    matriz_pesos = np.random.normal(scale=0.5, size=(len(vocabulario), dimensao)).astype(np.float32)
    matriz_pesos[0] = np.zeros(dimensao, dtype=np.float32) # O <PAD> (indice 0) fica zerado
    
    palavras_mapeadas = 0
    for palavra, indice in vocabulario.items():
        if palavra in modelo_w2v.wv:
            matriz_pesos[indice] = modelo_w2v.wv[palavra]
            palavras_mapeadas += 1
            
    print(f"  [Word2Vec Interno] Matriz pronta: {palavras_mapeadas:,} / {len(vocabulario):,} palavras mapeadas.")
    return torch.tensor(matriz_pesos)


if _TORCH_OK:
    class BiLSTMAtencaoNativa(nn.Module):
        """
        Rede Profunda: Matriz Word2Vec (Injetada) -> BiLSTM (2 Camadas) -> Atencao -> Saida
        """
        def __init__(
            self,
            matriz_pesos: torch.Tensor,
            dimensao_oculta: int = 256,
            quantidade_classes: int = 5,
            dropout: float = 0.4,
            n_camadas: int = 2
        ) -> None:
            super().__init__()
            
            self.embedding = nn.Embedding.from_pretrained(matriz_pesos, freeze=False, padding_idx=0)
            dim_embedding = matriz_pesos.shape[1]
            
            self.bilstm = nn.LSTM(
                input_size=dim_embedding,
                hidden_size=dimensao_oculta,
                num_layers=n_camadas,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_camadas > 1 else 0.0,
            )
            
            self.atencao = nn.Linear(dimensao_oculta * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.classificador = nn.Linear(dimensao_oculta * 2, quantidade_classes)

        def forward(self, input_ids):
            emb = self.dropout(self.embedding(input_ids))
            lstm_out, _ = self.bilstm(emb)
            
            pesos_atencao = torch.softmax(self.atencao(lstm_out), dim=1)
            representacao_doc = (lstm_out * pesos_atencao).sum(dim=1)
            
            return self.classificador(self.dropout(representacao_doc))


def treinar_bilstm_autossuficiente(
    textos_treino: list[str],
    y_treino: np.ndarray,
    textos_val: list[str],
    y_val: np.ndarray,
    textos_extras_w2v: list[str] | None = None,
    dim_embedding: int = 128,
    max_vocab: int = 30000,
    comprimento_maximo: int = 500,
    batch_size: int = 64,
    epocas: int = 15,
    lr: float = 1e-3,
    paciencia_early_stopping: int = 3
):
    """
    Treinador mestre: acopla os textos -1 no Word2Vec, balanceia a Loss e salva o pico de F1-Macro.
    """
    if not (_TORCH_OK and _GENSIM_OK):
        raise ImportError("PyTorch ou Gensim ausentes no ambiente.")
        
    # Constroi o vocabulario numerico
    vocabulario = construir_vocabulario(textos_treino, max_vocabulario=max_vocab, minimo_frequencia=2)
    
    # Ger matriz interna com os textos de treino + os textos -1
    matriz_w2v = gerar_matriz_embedding_interna(
        textos_treino=textos_treino, 
        vocabulario=vocabulario, 
        dimensao=dim_embedding,
        textos_extras_w2v=textos_extras_w2v
    )
    
    # Converte os textos em Datasets PyTorch
    ds_tr = DatasetSequencias(textos_treino, y_treino, vocabulario, comprimento_maximo)
    ds_vl = DatasetSequencias(textos_val, y_val, vocabulario, comprimento_maximo)
    
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False)
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    print(f"\n[Motor BiLSTM] Rodando no hardware: {dispositivo}")
    
    modelo = BiLSTMAtencaoNativa(matriz_pesos=matriz_w2v, dimensao_oculta=256).to(dispositivo)
    
    # Pesos para classes (arrumar o desbalancemaneto)
    contagem = Counter(y_treino.tolist())
    total_amostras = len(y_treino)
    n_classes = len(MAPEAMENTO_CLASSES)
    pesos_por_classe = [(total_amostras / (n_classes * max(contagem.get(i, 1), 1))) for i in range(n_classes)]
    
    tensor_pesos = torch.tensor(pesos_por_classe, dtype=torch.float).to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=tensor_pesos)
    
    otimizador = optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode='max', patience=1, factor=0.5)
    
    melhor_f1 = -1.0
    melhor_estado = None
    historico = []
    sem_melhorar = 0
    
    for epoca in range(epocas):
        modelo.train()
        perdas_treino = []
        for lote in loader_tr:
            x, y = lote["input_ids"].to(dispositivo), lote["labels"].to(dispositivo)
            
            otimizador.zero_grad()
            logits = modelo(x)
            perda = criterio(logits, y)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perdas_treino.append(perda.item())
            
        modelo.eval()
        preds_val, rots_val = [], []
        with torch.no_grad():
            for lote in loader_vl:
                x = lote["input_ids"].to(dispositivo)
                logits = modelo(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_val.extend(preds)
                rots_val.extend(lote["labels"].numpy())
                
        f1_atual = f1_score(rots_val, preds_val, average="macro")
        loss_media = float(np.mean(perdas_treino))
        scheduler.step(f1_atual)
        
        historico.append({"epoca": epoca+1, "loss_treino": loss_media, "f1_macro_val": f1_atual})
        print(f"    Epoca {epoca+1:02d}/{epocas} | Loss Treino: {loss_media:.4f} | F1-Macro Val: {f1_atual:.4f}")
        
        if f1_atual > melhor_f1:
            melhor_f1 = f1_atual
            melhor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            sem_melhorar = 0
        else:
            sem_melhorar += 1
            
        if sem_melhorar >= paciencia_early_stopping:
            print(f"  [Early Stopping] Rede sem evolucao. Treino interrompido na epoca {epoca+1}.")
            break
            
    modelo.load_state_dict(melhor_estado)
    print(f"--- TREINO CONCLUIDO | Pico de F1-Macro Val (BiLSTM Nativa): {melhor_f1:.4f} ---")
    
    return modelo, vocabulario, pd.DataFrame(historico)

if _TORCH_OK:
    class DatasetInferencia(Dataset):
        """
        Alimentador de dados cego: entrega apenas os textos (convertidos em números) para processar
        """
        def __init__(self, textos: list[str], vocabulario: dict[str, int], comprimento_maximo: int):
            self.textos = textos
            self.vocabulario = vocabulario
            self.comprimento_maximo = comprimento_maximo

        def __len__(self):
            return len(self.textos)

        def __getitem__(self, idx):
            texto_fatiado = str(self.textos[idx]).split()
            
            # Converte a palavra string no ID numérico do nosso dicionário. 
            # Se a palavra for inédita (não está no dicionário), recebe 0 (o vetor nulo <UNK>/<PAD>)
            indices = [self.vocabulario.get(palavra, 0) for palavra in texto_fatiado]
            
            # Corta se for maior que 500; preenche com zeros no final se for menor que 500
            if len(indices) < self.comprimento_maximo:
                indices = indices + [0] * (self.comprimento_maximo - len(indices))
            else:
                indices = indices[:self.comprimento_maximo]
                
            return torch.tensor(indices, dtype=torch.long)


def gerar_submissao_bilstm(
    modelo: torch.nn.Module, 
    vocabulario: dict[str, int], 
    df_teste: pd.DataFrame, 
    coluna_id: str = "Id",
    coluna_texto: str = "texto_limpo",
    comprimento_maximo: int = 500,
    batch_size: int = 64,
    nome_ficheiro_csv: str = "submissao_bilstm.csv"
) -> pd.DataFrame:
    """
    Desliga os motores de treino, passa as petições inéditas pela rede e salva o .csv final.
    """
    print(f"  Prepararanda as {len(df_teste):,} de teste para a BiLSTM...")
    
    # Monta a linha de montagem de dados para o teste
    textos_teste = df_teste[coluna_texto].tolist()
    ds_teste = DatasetInferencia(textos_teste, vocabulario, comprimento_maximo)
    loader_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False)
    
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    
    modelo.eval()
    
    predicoes_finais = []
    
    with torch.no_grad():
        for lote_x in loader_teste:
            x = lote_x.to(dispositivo)
            logits = modelo(x)
            classes_vencedoras = torch.argmax(logits, dim=1).cpu().numpy()
            predicoes_finais.extend(classes_vencedoras)
            
    df_submissao = pd.DataFrame({
        "Id": df_teste[coluna_id],
        "Category": predicoes_finais
    })
    
    df_submissao.to_csv(nome_ficheiro_csv, index=False)
    print(f"  [Sucesso] '{nome_ficheiro_csv}' guardado com {len(df_submissao)} predições!")
    
    return df_submissao


# =========================================================

class DatasetBERTUnificado(Dataset):
    """
    Alimentador Sênior: Realiza a amputação Head + Tail estritamente a nível de Subwords (WordPiece),
    garantindo que a decisão final do juiz nunca seja descartada pelo limite de 512.
    """
    def __init__(self, textos: list[str], rotulos: Iterable | None, tokenizer, max_len: int = 512):
        self.textos = textos
        self.rotulos = list(rotulos) if rotulos is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = str(self.textos[idx])
        
        # Tokenizamos o texto inteiro em IDs brutos (SEM truncar, SEM special tokens)
        ids_brutos = self.tokenizer.encode(texto, add_special_tokens=False)
        
        # Reserva 2 espaços obrigatórios para [CLS] e [SEP] -> sobram 510 vagas dinâmicas
        vagas = self.max_len - 2
        
        if len(ids_brutos) > vagas:
            # HEAD + TAIL DE SUBWORDS JURÍDICAS
            # Agarra 128 subwords da frente (Petição inicial) e as últimas 382 subwords do final absoluto (Veredito)
            head = ids_brutos[:128]
            tail = ids_brutos[-(vagas - 128):]
            input_ids = [self.tokenizer.cls_token_id] + head + tail + [self.tokenizer.sep_token_id]
        else:
            input_ids = [self.tokenizer.cls_token_id] + ids_brutos + [self.tokenizer.sep_token_id]
            
        # Preenchimento manual de zeros (Padding) e montagem da Attention Mask
        tamanho_real = len(input_ids)
        pad_falta = self.max_len - tamanho_real
        
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_falta
        attention_mask = [1] * tamanho_real + [0] * pad_falta
        
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        if self.rotulos is not None:
            item['labels'] = torch.tensor(int(self.rotulos[idx]), dtype=torch.long)
            
        return item


def treinar_legal_bert_otimizado(
    textos_treino: list[str],
    y_treino: Iterable,
    textos_val: list[str],
    y_val: Iterable,
    nome_modelo: str = "rufimelo/Legal-BERTimbau-base",
    max_len: int = 512,
    batch_size: int = 16,       
    acumulo_grad: int = 2,      
    epocas: int = 4,            
    lr: float = 2e-5
):
    print(f"  [Legal-BERT] Instanciando a arquitetura '{nome_modelo}'...")
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(nome_modelo, num_labels=5)

    ds_tr = DatasetBERTUnificado(textos_treino, y_treino, tokenizer, max_len)
    ds_vl = DatasetBERTUnificado(textos_val, y_val, tokenizer, max_len)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    print(f"  [Legal-BERT] Motor alocado no hardware: {dispositivo}")
    modelo = modelo.to(dispositivo)

    lista_y_treino = [int(r) for r in y_treino]
    contagem = Counter(lista_y_treino)
    total_amostras = len(lista_y_treino)
    n_classes = len(set(lista_y_treino))
    
    pesos = [total_amostras / (n_classes * max(contagem.get(i, 1), 1)) for i in range(n_classes)]
    tensor_pesos = torch.tensor(pesos, dtype=torch.float).to(dispositivo)
    criterio = torch.nn.CrossEntropyLoss(weight=tensor_pesos)

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    passos_totais = (len(loader_tr) // acumulo_grad) * epocas
    scheduler = get_linear_schedule_with_warmup(otimizador, num_warmup_steps=int(passos_totais * 0.1), num_training_steps=passos_totais)

    scaler = torch.cuda.amp.GradScaler()
    melhor_f1 = -1.0
    melhor_estado = None
    historico = []

    for epoca in range(epocas):
        modelo.train()
        perdas_epoca = []
        otimizador.zero_grad()

        for passo, lote in enumerate(loader_tr):
            b_ids = lote['input_ids'].to(dispositivo)
            b_mask = lote['attention_mask'].to(dispositivo)
            b_labels = lote['labels'].to(dispositivo)

            with torch.amp.autocast("cuda"):
                saida = modelo(input_ids=b_ids, attention_mask=b_mask)
                perda = criterio(saida.logits, b_labels) / acumulo_grad

            scaler.scale(perda).backward()
            perdas_epoca.append(perda.item() * acumulo_grad)

            if (passo + 1) % acumulo_grad == 0 or (passo + 1) == len(loader_tr):
                scaler.unscale_(otimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                scaler.step(otimizador)
                scaler.update()
                scheduler.step()
                otimizador.zero_grad()

        modelo.eval()
        preds_val, rots_val = [], []
        with torch.no_grad():
            for lote_vl in loader_vl:
                b_ids = lote_vl['input_ids'].to(dispositivo)
                b_mask = lote_vl['attention_mask'].to(dispositivo)

                with torch.amp.autocast("cuda"):
                    logits = modelo(input_ids=b_ids, attention_mask=b_mask).logits
                    
                preds_val.extend(torch.argmax(logits, dim=1).cpu().numpy())
                rots_val.extend(lote_vl['labels'].cpu().numpy())

        f1_atual = f1_score(rots_val, preds_val, average="macro")
        loss_m = float(np.mean(perdas_epoca))
        historico.append({"epoca": epoca+1, "loss_treino": loss_m, "f1_macro_val": f1_atual})

        print(f"    Época {epoca+1:02d}/{epocas} | Loss Treino: {loss_m:.4f} | F1-Macro Val: {f1_atual:.4f}")

        if f1_atual > melhor_f1:
            melhor_f1 = f1_atual
            melhor_estado = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}

    modelo.load_state_dict(melhor_estado)
    print(f"--- FINE-TUNING JURÍDICO CONCLUÍDO | Pico de F1-Macro Val (Legal-BERT): {melhor_f1:.4f} ---")
    return modelo, tokenizer, pd.DataFrame(historico)


def gerar_submissao_legal_bert(
    modelo, 
    tokenizer, 
    df_teste: pd.DataFrame, 
    coluna_id: str = "Id", 
    coluna_texto: str = "texto_limpo",
    max_len: int = 512,
    batch_size: int = 32, 
    nome_ficheiro_csv: str = "submissao_legal_bert_v1.csv"
) -> pd.DataFrame:
    print(f"  [Legal-BERT Inferência] Lendo {len(df_teste):,} processos inéditos de teste...")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(dispositivo)
    modelo.eval()

    ds_teste = DatasetBERTUnificado(df_teste[coluna_texto].tolist(), rotulos=None, tokenizer=tokenizer, max_len=max_len)
    loader_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False)

    preds_finais = []
    with torch.no_grad():
        for lote in loader_teste:
            b_ids = lote['input_ids'].to(dispositivo)
            b_mask = lote['attention_mask'].to(dispositivo)

            with torch.amp.autocast("cuda"):
                logits = modelo(input_ids=b_ids, attention_mask=b_mask).logits

            preds_finais.extend(torch.argmax(logits, dim=1).cpu().numpy())

    df_submissao = pd.DataFrame({
        "Id": df_teste[coluna_id],
        "Category": preds_finais
    })
    df_submissao.to_csv(nome_ficheiro_csv, index=False)
    print(f"  [Sucesso] Ficheiro gerado: '{nome_ficheiro_csv}'")
    return df_submissao
