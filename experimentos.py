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


# ==============================================================
# MODELO PROFUNDO - BiLSTM COM ATENCAO
# ==============================================================

class TokenizadorBiLSTM:
    """
    Tokenizador simples para o BiLSTM:
    constroi um vocabulario a partir do corpus e codifica sequencias como IDs.
    Tokens ausentes recebem o indice <UNK> = 1.
    """

    def __init__(self, max_vocab: int = 30000, max_len: int = 300) -> None:
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def construir_vocabulario(self, corpus: list[list[str]]) -> None:
        """Constroi vocabulario a partir de lista de listas de tokens."""
        contador: Counter = Counter()
        for tokens in corpus:
            for token in tokens:
                contador[token] += 1
        for palavra, _ in contador.most_common(self.max_vocab - 2):
            if palavra not in self.vocab:
                self.vocab[palavra] = len(self.vocab)
        print(f"  Vocabulario BiLSTM: {len(self.vocab):,} tokens (max_vocab={self.max_vocab})")

    def codificar(self, tokens: list[str]) -> list[int]:
        """Codifica lista de tokens em sequencia de IDs com padding e truncamento."""
        ids = []
        for token in tokens[: self.max_len]:
            ids.append(self.vocab.get(token, 1))  # 1 = <UNK>
        while len(ids) < self.max_len:
            ids.append(0)  # 0 = <PAD>
        return ids

    def codificar_batch(self, lista_tokens: list[list[str]]) -> list[list[int]]:
        """Codifica batch de listas de tokens em batch de IDs."""
        resultado = []
        for tokens in lista_tokens:
            resultado.append(self.codificar(tokens))
        return resultado


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

    class BiLSTMAtencao(nn.Module):
        """
        BiLSTM com mecanismo de atencao para classificacao de texto.

        Arquitetura:
          Embedding -> Dropout -> BiLSTM (2 camadas) -> Atencao -> Linear

        - Bidirecional: processa cada token com contexto anterior e posterior.
        - Atencao: foca nos tokens mais discriminativos de cada documento,
          util para textos juridicos longos com informacao distribuida.
        - 2 camadas: maior capacidade de representacao hierarquica.
        - Dropout: regularizacao para evitar overfitting.
        """

        def __init__(
            self,
            tamanho_vocabulario: int,
            dimensao_embedding: int = 128,
            dimensao_oculta: int = 256,
            quantidade_classes: int = 5,
            dropout: float = 0.4,
            n_camadas: int = 2,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                tamanho_vocabulario, dimensao_embedding, padding_idx=0
            )
            self.bilstm = nn.LSTM(
                input_size=dimensao_embedding,
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
            emb = self.dropout(self.embedding(input_ids))       # (B, T, E)
            lstm_out, _ = self.bilstm(emb)                      # (B, T, 2H)
            pesos = torch.softmax(self.atencao(lstm_out), dim=1)  # (B, T, 1)
            contexto = (lstm_out * pesos).sum(dim=1)             # (B, 2H)
            return self.classificador(self.dropout(contexto))    # (B, C)

else:
    # Stubs para quando PyTorch nao esta instalado
    class DatasetSequencias:  # type: ignore
        pass

    class BiLSTMAtencao:  # type: ignore
        pass


def treinar_bilstm(
    textos_treino: Iterable[str],
    y_treino,
    textos_validacao: Iterable[str],
    y_validacao,
    max_vocabulario: int = 30000,
    comprimento_max: int = 300,
    epocas: int = 10,
    tamanho_lote: int = 64,
    taxa_aprendizagem: float = 1e-3,
    dropout: float = 0.4,
    dim_embedding: int = 128,
    dim_oculto: int = 256,
    num_camadas: int = 2,
    random_state: int = 42,
):
    """
    Treina o BiLSTM com atencao. Melhorias sobre BiLSTM simples:
    - CrossEntropyLoss ponderada: compensa o desbalanceamento de classes
    - Gradient clipping (max_norm=1.0): estabiliza treinamento de LSTMs profundas
    - ReduceLROnPlateau: reduz lr se F1-Val parar de melhorar (patience=2)
    - Salva melhor checkpoint por F1-Macro de validacao
    Retorna: (modelo, vocabulario, DataFrame com historico de treinamento)
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch nao instalado. Execute: pip install torch")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    textos_treino_list = list(textos_treino)
    textos_validacao_list = list(textos_validacao)
    y_treino_arr = np.asarray(list(y_treino))
    y_validacao_arr = np.asarray(list(y_validacao))

    vocabulario = construir_vocabulario(textos_treino_list, max_vocabulario=max_vocabulario)

    dataset_treino = DatasetSequencias(
        textos_treino_list, y_treino_arr, vocabulario, comprimento_max
    )
    dataset_validacao = DatasetSequencias(
        textos_validacao_list, y_validacao_arr, vocabulario, comprimento_max
    )

    loader_treino = DataLoader(dataset_treino, batch_size=tamanho_lote, shuffle=True)
    loader_validacao = DataLoader(dataset_validacao, batch_size=tamanho_lote, shuffle=False)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {dispositivo}")

    modelo = BiLSTMAtencao(
        tamanho_vocabulario=len(vocabulario),
        dimensao_embedding=dim_embedding,
        dimensao_oculta=dim_oculto,
        n_camadas=num_camadas,
        dropout=dropout,
    ).to(dispositivo)

    # Pesos inversamente proporcionais a frequencia de cada classe
    contagem_classes = Counter(y_treino_arr.tolist())
    n_total = len(y_treino_arr)
    n_classes = len(MAPEAMENTO_CLASSES)
    pesos = torch.tensor(
        [n_total / (n_classes * max(contagem_classes.get(i, 1), 1)) for i in range(n_classes)],
        dtype=torch.float,
    ).to(dispositivo)

    criterio = nn.CrossEntropyLoss(weight=pesos)
    otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizagem, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", patience=2, factor=0.5
    )

    melhor_f1 = -1.0
    melhor_estado = None
    historico = []

    for epoca in range(epocas):
        modelo.train()
        perdas_treino = []
        for lote in loader_treino:
            entradas = lote["input_ids"].to(dispositivo)
            rotulos = lote["labels"].to(dispositivo)

            otimizador.zero_grad()
            logits = modelo(entradas)
            perda = criterio(logits, rotulos)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perdas_treino.append(float(perda.item()))

        modelo.eval()
        predicoes_val = []
        rotulos_val = []
        with torch.no_grad():
            for lote in loader_validacao:
                entradas = lote["input_ids"].to(dispositivo)
                rots = lote["labels"].cpu().numpy()
                logits = modelo(entradas)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predicoes_val.extend(preds.tolist())
                rotulos_val.extend(rots.tolist())

        f1_macro = f1_score(rotulos_val, predicoes_val, average="macro")
        loss_media = float(np.mean(perdas_treino))
        scheduler.step(f1_macro)

        historico.append({
            "epoca": epoca + 1,
            "loss_treino": loss_media,
            "f1_macro_validacao": float(f1_macro),
        })
        print(
            f"  Epoca {epoca + 1:02d}/{epocas} | "
            f"Loss: {loss_media:.4f} | F1-Val: {f1_macro:.4f}"
        )

        if f1_macro > melhor_f1:
            melhor_f1 = f1_macro
            melhor_estado = {
                k: v.detach().cpu() for k, v in modelo.state_dict().items()
            }

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    print(f"  Melhor F1-Val (BiLSTM): {melhor_f1:.4f}")
    return modelo, vocabulario, pd.DataFrame(historico)


def predizer_bilstm(
    modelo,
    textos: Iterable[str],
    vocabulario: dict[str, int],
    comprimento_max: int = 300,
    batch_size: int = 128,
) -> np.ndarray:
    """Gera predicoes do BiLSTM em modo de inferencia."""
    if not _TORCH_OK:
        raise ImportError("PyTorch nao instalado.")

    dataset = DatasetSequencias(
        textos=list(textos),
        rotulos=None,
        vocabulario=vocabulario,
        comprimento_maximo=comprimento_max,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dispositivo = next(modelo.parameters()).device
    modelo.eval()

    predicoes = []
    with torch.no_grad():
        for lote in loader:
            entradas = lote["input_ids"].to(dispositivo)
            logits = modelo(entradas)
            predicoes.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return np.asarray(predicoes)


# ==============================================================
# TRANSFORMERS - BERTimbau (fine-tuning)
# ==============================================================

if _TORCH_OK:

    class DatasetTransformer(Dataset):
        """Dataset PyTorch compativel com HuggingFace Trainer."""

        def __init__(self, encodings, labels=None) -> None:
            self.encodings = encodings
            self.labels = labels

        def __len__(self) -> int:
            return len(self.encodings["input_ids"])

        def __getitem__(self, indice: int):
            item = {
                chave: torch.tensor(valor[indice], dtype=torch.long)
                for chave, valor in self.encodings.items()
            }
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[indice], dtype=torch.long)
            return item

else:
    class DatasetTransformer:  # type: ignore
        pass


def treinar_transformer(
    textos_treino: Iterable[str],
    y_treino,
    textos_validacao: Iterable[str],
    y_validacao,
    nome_modelo: str = "neuralmind/bert-base-portuguese-cased",
    diretorio_saida: str = "resultados_transformer",
    comprimento_maximo: int = 256,
    comprimento_max: int | None = None,
    batch_size: int = 16,
    tamanho_lote: int | None = None,
    epocas: int = 3,
    num_epocas: int | None = None,
    taxa_aprendizado: float = 2e-5,
    taxa_aprendizagem: float | None = None,
):
    """
    Fine-tuning do BERTimbau para classificacao de documentos juridicos.

    Configuracoes de treinamento:
    - warmup_ratio=0.1: 10% dos steps com warmup linear do lr
    - weight_decay=0.01: regularizacao L2 para evitar overfitting
    - fp16: precisao mista em GPU (acelera treinamento ~2x)
    - EarlyStoppingCallback: interrompe se F1-Macro nao melhorar em 2 epochs
    - load_best_model_at_end: restaura melhor checkpoint automaticamente

    Retorna: (trainer, tokenizador)
    """
    if not (_TRANSFORMERS_OK and _TORCH_OK):
        raise ImportError("transformers e/ou torch nao instalados.")

    import os
    os.makedirs(diretorio_saida, exist_ok=True)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dispositivo: {dispositivo} | Modelo: {nome_modelo}")

    if num_epocas is not None:
        epocas = num_epocas
    if tamanho_lote is not None:
        batch_size = tamanho_lote
    if comprimento_max is not None:
        comprimento_maximo = comprimento_max
    if taxa_aprendizagem is not None:
        taxa_aprendizado = taxa_aprendizagem

    tokenizador = AutoTokenizer.from_pretrained(nome_modelo)
    modelo_bert = AutoModelForSequenceClassification.from_pretrained(
        nome_modelo, num_labels=len(MAPEAMENTO_CLASSES)
    )

    def _tokenizar(textos):
        return tokenizador(
            list(textos), max_length=comprimento_maximo,
            truncation=True, padding="max_length",
        )

    print("  Tokenizando treino e validacao...")
    enc_treino = _tokenizar(textos_treino)
    enc_validacao = _tokenizar(textos_validacao)

    ds_treino = DatasetTransformer(enc_treino, list(y_treino))
    ds_validacao = DatasetTransformer(enc_validacao, list(y_validacao))

    def calcular_metricas(eval_pred):
        logits, labels = eval_pred
        predicoes = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predicoes)),
            "f1_macro": float(f1_score(labels, predicoes, average="macro")),
        }

    # Compatibilidade com versoes antigas e novas do transformers
    argumentos_base = dict(
        output_dir=diretorio_saida,
        learning_rate=taxa_aprendizado,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epocas,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        seed=42,
        fp16=(dispositivo == "cuda"),
        report_to="none",
    )

    try:
        argumentos = TrainingArguments(
            eval_strategy="epoch", save_strategy="epoch", **argumentos_base
        )
    except TypeError:
        try:
            argumentos = TrainingArguments(
                evaluation_strategy="epoch", save_strategy="epoch", **argumentos_base
            )
        except Exception:
            argumentos = TrainingArguments(**argumentos_base)

    callbacks = []
    if _TRANSFORMERS_OK:
        try:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        except Exception:
            pass

    # Suporte a 'processing_class' (novo) ou 'tokenizer' (legado)
    try:
        trainer = Trainer(
            model=modelo_bert,
            args=argumentos,
            train_dataset=ds_treino,
            eval_dataset=ds_validacao,
            processing_class=tokenizador,
            compute_metrics=calcular_metricas,
            callbacks=callbacks,
        )
    except TypeError:
        trainer = Trainer(
            model=modelo_bert,
            args=argumentos,
            train_dataset=ds_treino,
            eval_dataset=ds_validacao,
            tokenizer=tokenizador,
            compute_metrics=calcular_metricas,
            callbacks=callbacks,
        )

    print("  Iniciando fine-tuning do BERTimbau...")
    trainer.train()
    return trainer, tokenizador


def predizer_transformer(
    trainer,
    tokenizador,
    textos: Iterable[str],
    comprimento_maximo: int = 256,
) -> np.ndarray:
    """Gera predicoes do transformer em modo de inferencia."""
    encodings = tokenizador(
        list(textos),
        truncation=True,
        padding="max_length",
        max_length=comprimento_maximo,
    )
    dataset = DatasetTransformer(encodings, labels=None)
    resultado = trainer.predict(dataset)
    return np.argmax(resultado.predictions, axis=-1)


# ==============================================================
# PREPARACAO PARA INFERENCIA (modelos classicos)
# ==============================================================

def preparar_predicao(
    textos: Iterable[str],
    representacao: str,
    artefato_representacao,
) -> np.ndarray:
    """
    Transforma textos para inferencia com o artefato de representacao salvo.
    Suporta TF-IDF (esparso), Word2Vec (denso) e Naive Bayes (TF-IDF + scaler).
    """
    if representacao == "TF-IDF":
        if isinstance(artefato_representacao, tuple):
            scaler, vetorizador = artefato_representacao
            return scaler.transform(vetorizador.transform(list(textos)))
        return artefato_representacao.transform(list(textos))
    if representacao == "Word2Vec":
        return vetorizar_word2vec_media(textos, artefato_representacao)
    raise ValueError(f"Representacao nao suportada: {representacao}")


def ajustar_modelo_final(
    nome_modelo: str,
    representacao: str,
    textos_treino: Iterable[str],
    y_treino,
):
    """
    Treina o modelo selecionado com todos os dados de treino (sem validacao).
    Usado para gerar a submissao final apos selecionar o melhor modelo.
    """
    textos_treino_list = list(textos_treino)
    y_treino_arr = np.asarray(list(y_treino))

    if representacao == "TF-IDF":
        vet, X_treino = vetorizar_tfidf(textos_treino_list)
        if "Naive Bayes" in nome_modelo:
            scaler = MaxAbsScaler()
            X_treino = scaler.fit_transform(X_treino)
            modelo = treinar_naive_bayes(X_treino, y_treino_arr)
            return modelo, (scaler, vet)
        elif "Regressao Logistica" in nome_modelo:
            modelo = treinar_regressao_logistica(X_treino, y_treino_arr)
        elif "SVM" in nome_modelo:
            modelo = treinar_svm_linear(X_treino, y_treino_arr)
        else:
            raise ValueError(f"Modelo nao suportado: {nome_modelo}")
        return modelo, vet

    if representacao == "Word2Vec":
        modelo_w2v = treinar_word2vec(textos_treino_list)
        X_treino = vetorizar_word2vec_media(textos_treino_list, modelo_w2v)
        modelo = treinar_regressao_logistica(X_treino, y_treino_arr)
        return modelo, modelo_w2v

    raise ValueError(f"Representacao nao suportada: {representacao}")


__all__ = [
    "MAPEAMENTO_CLASSES",
    "ResultadoExperimento",
    "dividir_treino_validacao",
    "criar_vetorizador_tfidf",
    "vetorizar_tfidf",
    "tokenizar_para_word2vec",
    "treinar_word2vec",
    "vetorizar_word2vec_media",
    "treinar_regressao_logistica",
    "treinar_svm_linear",
    "treinar_naive_bayes",
    "avaliar_modelo_em_validacao",
    "avaliar_validacao_cruzada",
    "executar_baselines_classicos",
    "aplicar_ner_juridico",
    "extrair_mencoes_legais_amostra",
    "TokenizadorBiLSTM",
    "construir_vocabulario",
    "codificar_texto",
    "DatasetSequencias",
    "BiLSTMAtencao",
    "treinar_bilstm",
    "predizer_bilstm",
    "DatasetTransformer",
    "treinar_transformer",
    "predizer_transformer",
    "preparar_predicao",
    "ajustar_modelo_final",
]
