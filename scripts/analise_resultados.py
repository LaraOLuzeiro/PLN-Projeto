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
Modulo de Analise dos Resultados.

Contem funcoes para:
  - Calculo de metricas de classificacao (F1-Macro, F1 por classe, Acuracia, etc.)
  - Plotagem de matrizes de confusao (absoluta e normalizada)
  - Comparacao tabular e grafica de multiplos modelos
  - Curvas de treinamento (loss e F1-Val por epoca)
  - Teste estatistico de McNemar para comparar dois classificadores
  - Analise de erros (exemplos mal classificados)
    - Avaliacao integrada de modelos classicos e profundos
    - Inferencia por probabilidades (BiLSTM, Legal-BERT, Ensemble)
  - Geracao de arquivo submission.csv para a competicao Kaggle
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MAPEAMENTO_CLASSES = {
    0: "Acordao",
    1: "ARE",
    2: "Despacho",
    3: "RE",
    4: "Sentenca",
}

NOMES_CLASSES = list(MAPEAMENTO_CLASSES.values())


# ==============================================================
# METRICAS DE CLASSIFICACAO
# ==============================================================

def calcular_metricas_classificacao(
    y_verdadeiro,
    y_predito,
    nome_modelo: str = "Modelo",
) -> dict:
    """
    Calcula metricas robustas ao desbalanceamento.
    F1-Macro e a metrica principal: trata todas as classes com igual peso,
    penalizando igualmente erros em classes raras (Acordao, Despacho) e
    nas mais frequentes (RE). Inclui F1 por classe para analise detalhada.
    """
    rotulos_ordenados = sorted(MAPEAMENTO_CLASSES)
    resultado = {
        "modelo": nome_modelo,
        "accuracy": float(accuracy_score(y_verdadeiro, y_predito)),
        "f1_macro": float(
            f1_score(
                y_verdadeiro,
                y_predito,
                labels=rotulos_ordenados,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_verdadeiro,
                y_predito,
                labels=rotulos_ordenados,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_micro": float(
            f1_score(
                y_verdadeiro,
                y_predito,
                labels=rotulos_ordenados,
                average="micro",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_verdadeiro,
                y_predito,
                labels=rotulos_ordenados,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_verdadeiro,
                y_predito,
                labels=rotulos_ordenados,
                average="macro",
                zero_division=0,
            )
        ),
    }
    # F1 por classe (granularidade para analise por tipo de documento)
    f1_por_classe = f1_score(
        y_verdadeiro,
        y_predito,
        labels=rotulos_ordenados,
        average=None,
        zero_division=0,
    )
    for i, nome in enumerate(NOMES_CLASSES):
        resultado[f"f1_{nome}"] = float(f1_por_classe[i]) if i < len(f1_por_classe) else 0.0

    return resultado


# ==============================================================
# MATRIZ DE CONFUSAO
# ==============================================================

def plotar_matriz_confusao(
    y_verdadeiro=None,
    y_predito=None,
    nome_modelo: str = "Modelo",
    matriz_confusao: pd.DataFrame | None = None,
    salvar_em: str | None = None,
) -> None:
    """
    Plota matrizes de confusao absolutas e normalizadas lado a lado.
    Versao normalizada mostra recall por classe (proporcao de acertos por classe real),
    util para identificar quais classes tem maior taxa de erro.
    Aceita (y_verdadeiro, y_predito) ou matriz pre-calculada.
    """
    os.makedirs("figs", exist_ok=True)

    if matriz_confusao is not None:
        cm_abs = matriz_confusao.values.astype(int)
        nomes = list(matriz_confusao.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            totais = cm_abs.sum(axis=1, keepdims=True)
            cm_norm = np.where(totais > 0, cm_abs / totais, 0.0)
    else:
        rotulos = sorted(MAPEAMENTO_CLASSES)
        nomes = NOMES_CLASSES
        cm_abs = confusion_matrix(y_verdadeiro, y_predito, labels=rotulos)
        cm_norm = confusion_matrix(y_verdadeiro, y_predito, labels=rotulos, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cm, titulo, fmt in zip(
        axes,
        [cm_abs, cm_norm],
        ["Contagem Absoluta", "Normalizada (recall por classe)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=nomes, yticklabels=nomes,
            ax=ax, linewidths=0.4,
        )
        ax.set_title(f"{nome_modelo} — {titulo}", fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if salvar_em is None:
        nome_arquivo = re.sub(r"[^a-zA-Z0-9_]", "_", nome_modelo).lower()
        salvar_em = f"figs/mc_{nome_arquivo}.png"
    plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# COMPARACAO DE MODELOS
# ==============================================================

def comparar_modelos(lista_resultados: list[dict]) -> pd.DataFrame:
    """
    Recebe lista de dicionarios retornados por calcular_metricas_classificacao().
    Retorna DataFrame comparativo ordenado por F1-Macro com exibicao formatada.
    """
    df = pd.DataFrame(lista_resultados)
    if "modelo" in df.columns:
        df = df.set_index("modelo")
    df = df.sort_values("f1_macro", ascending=False)

    cols_meta = ["familia", "protocolo_validacao", "n_amostras_validacao"]
    cols_princ = ["f1_macro", "f1_weighted", "accuracy", "precision_macro", "recall_macro"]
    cols_disp = [c for c in (cols_meta + cols_princ) if c in df.columns]

    print("\nComparacao de Modelos (ordenado por F1-Macro):")
    print("=" * 70)
    print(df[cols_disp].round(4).to_string())
    return df


def plotar_comparacao_modelos(
    df_comparacao: pd.DataFrame,
    salvar_em: str | None = "figs/comparacao_modelos.png",
) -> None:
    """
    Plota dois graficos de comparacao:
    1) Barras agrupadas: F1-Macro, F1-Weighted e Acuracia por modelo.
    2) Barras agrupadas: F1-Score por classe e modelo.
    Linha tracejada vermelha indica o threshold de 0.80.
    """
    os.makedirs("figs", exist_ok=True)
    metricas_gerais = ["f1_macro", "f1_weighted", "accuracy"]
    metricas_disp = [m for m in metricas_gerais if m in df_comparacao.columns]

    modelos = df_comparacao.index.tolist()
    x = np.arange(len(modelos))
    largura = 0.25
    cores = ["#2196F3", "#4CAF50", "#FF9800"]
    labels_metricas = {
        "f1_macro": "F1 Macro",
        "f1_weighted": "F1 Weighted",
        "accuracy": "Acuracia",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Grafico 1: metricas gerais
    for i, metrica in enumerate(metricas_disp):
        axes[0].bar(
            x + i * largura,
            df_comparacao[metrica].values,
            largura,
            label=labels_metricas.get(metrica, metrica),
            color=cores[i],
            alpha=0.85,
        )
    axes[0].set_xticks(x + largura)
    axes[0].set_xticklabels(modelos, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Valor")
    axes[0].set_title("Metricas por Modelo", fontweight="bold")
    axes[0].legend()
    axes[0].axhline(y=0.8, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # Grafico 2: F1 por classe
    cols_classe = [
        c for c in df_comparacao.columns
        if c.startswith("f1_") and not any(s in c for s in ["macro", "micro", "weighted"])
    ]
    if cols_classe:
        df_f1 = df_comparacao[cols_classe].copy()
        df_f1.columns = [c.replace("f1_", "") for c in df_f1.columns]
        df_f1.T.plot(kind="bar", ax=axes[1], colormap="Set2", width=0.72, alpha=0.9)
        axes[1].set_title("F1-Score por Classe e Modelo", fontweight="bold")
        axes[1].set_xlabel("Classe")
        axes[1].set_ylabel("F1-Score")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if salvar_em:
        plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# CURVAS DE TREINAMENTO (modelos profundos)
# ==============================================================

def plotar_curvas_treinamento(
    historico: dict | pd.DataFrame,
    nome_modelo: str = "BiLSTM",
    salvar_em: str | None = None,
) -> None:
    """
    Plota loss de treino e F1-Macro de validacao ao longo das epocas.
    Essencial para diagnosticar overfitting e verificar convergencia.
    """
    os.makedirs("figs", exist_ok=True)

    if isinstance(historico, pd.DataFrame):
        loss_treino = historico["loss_treino"].tolist()
        f1_val = historico.get(
            "f1_macro_validacao",
            historico.get("f1_macro_val", historico.get("f1_val", pd.Series())),
        ).tolist()
    else:
        loss_treino = historico.get("loss_treino", [])
        f1_val = historico.get(
            "f1_macro_validacao",
            historico.get("f1_macro_val", historico.get("f1_val", [])),
        )

    epocas = range(1, len(loss_treino) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(epocas), loss_treino, "b-o", markersize=4)
    axes[0].set_title(f"{nome_modelo} — Loss de Treinamento", fontweight="bold")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if f1_val:
        axes[1].plot(list(epocas[:len(f1_val)]), list(f1_val), "g-o", markersize=4)
        axes[1].set_title(f"{nome_modelo} — F1-Macro Validacao", fontweight="bold")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("F1-Macro")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if salvar_em is None:
        nome_arq = re.sub(r"[^a-zA-Z0-9_]", "_", nome_modelo).lower()
        salvar_em = f"figs/treinamento_{nome_arq}.png"
    plt.savefig(salvar_em, dpi=120, bbox_inches="tight")
    plt.show()


# ==============================================================
# TESTE ESTATISTICO - McNemar
# ==============================================================

def teste_mcnemar(
    y_real,
    pred1,
    pred2,
    nome1: str = "Modelo1",
    nome2: str = "Modelo2",
) -> dict:
    """
    Teste de McNemar para comparacao estatistica entre dois classificadores.
    H0: os dois modelos nao diferem em suas probabilidades de erro.
    Rejeita H0 (p_valor < 0.05): diferenca estatisticamente significativa.

    b = casos em que modelo1 acerta e modelo2 erra
    c = casos em que modelo1 erra e modelo2 acerta

    Importante para o relatorio: justifica escolha do melhor modelo
    com base em evidencia estatistica, nao apenas na media de F1.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        print("scipy nao instalado. Execute: pip install scipy")
        return {}

    y_real = np.array(y_real)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    certo1 = (pred1 == y_real)
    certo2 = (pred2 == y_real)

    b = int(np.sum(certo1 & ~certo2))
    c = int(np.sum(~certo1 & certo2))

    tabela = np.array([
        [int(np.sum(certo1 & certo2)), b],
        [c, int(np.sum(~certo1 & ~certo2))],
    ])

    chi2, p_valor, _, _ = chi2_contingency(tabela, correction=True)

    print(f"\nTeste de McNemar: {nome1} vs {nome2}")
    print(f"  b ({nome1} certo, {nome2} errado): {b}")
    print(f"  c ({nome1} errado, {nome2} certo): {c}")
    print(f"  Chi2 = {chi2:.4f}  |  p-valor = {p_valor:.4f}")

    if p_valor < 0.05:
        vencedor = nome1 if b < c else nome2
        print(f"  Resultado: diferenca SIGNIFICATIVA (p < 0.05). {vencedor} e superior.")
    else:
        print(f"  Resultado: sem diferenca significativa (p >= 0.05).")

    return {"chi2": float(chi2), "p_valor": float(p_valor), "b": b, "c": c}


# ==============================================================
# ANALISE DE ERROS
# ==============================================================

def analisar_erros(
    df_val: pd.DataFrame,
    y_real,
    y_pred,
    coluna_texto: str = "texto_preprocessado",
    n: int = 10,
) -> None:
    """
    Exibe os primeiros n exemplos mal classificados.
    Util para identificar padroes de confusao entre classes e
    orientar ajustes no preprocessamento ou na arquitetura do modelo.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    erros_idx = np.where(y_real != y_pred)[0]

    print(f"\nAnalise de Erros (primeiros {min(n, len(erros_idx))} casos):")
    print("-" * 70)
    for i in erros_idx[:n]:
        real = MAPEAMENTO_CLASSES.get(int(y_real[i]), str(y_real[i]))
        previsto = MAPEAMENTO_CLASSES.get(int(y_pred[i]), str(y_pred[i]))
        texto = ""
        if coluna_texto in df_val.columns:
            texto = str(df_val.iloc[i][coluna_texto])[:120]
        print(f"  Real: {real:<12} | Previsto: {previsto:<12} | Texto: {texto}...")


# ==============================================================
# AUXILIARES PARA AVALIACAO INTEGRADA (CLASSICOS + PROFUNDOS)
# ==============================================================

def registrar_resultado_modelo(
    lista_resultados: list[dict],
    mapa_predicoes: dict[str, np.ndarray],
    mapa_y_verdadeiro: dict[str, np.ndarray],
    nome_modelo: str,
    y_verdadeiro,
    y_predito,
    familia: str,
    protocolo_validacao: str,
) -> dict:
    """
    Calcula metricas de um modelo, registra metadados de avaliacao e
    armazena y_true/y_pred para analises posteriores (matriz, McNemar, erros).
    """
    y_true_arr = np.asarray(y_verdadeiro)
    y_pred_arr = np.asarray(y_predito)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"Tamanhos incompativeis para '{nome_modelo}': "
            f"y_true={y_true_arr.shape[0]} vs y_pred={y_pred_arr.shape[0]}."
        )

    metricas = calcular_metricas_classificacao(
        y_true_arr,
        y_pred_arr,
        nome_modelo=nome_modelo,
    )
    metricas["familia"] = familia
    metricas["protocolo_validacao"] = protocolo_validacao
    metricas["n_amostras_validacao"] = int(y_true_arr.shape[0])

    lista_resultados.append(metricas)
    mapa_predicoes[nome_modelo] = y_pred_arr
    mapa_y_verdadeiro[nome_modelo] = y_true_arr
    return metricas


def selecionar_top_modelos(
    tabela_resultados: pd.DataFrame,
    n: int = 2,
    familia: str | None = None,
    protocolo_validacao: str | None = None,
) -> list[str]:
    """
    Seleciona os top-N modelos por F1-Macro (desempate por acuracia),
    com filtros opcionais por familia e protocolo.
    """
    if tabela_resultados.empty:
        return []

    df = tabela_resultados.copy()
    if "modelo" in df.columns:
        df = df.set_index("modelo")

    if familia is not None and "familia" in df.columns:
        df = df[df["familia"] == familia]
    if protocolo_validacao is not None and "protocolo_validacao" in df.columns:
        df = df[df["protocolo_validacao"] == protocolo_validacao]

    if df.empty:
        return []

    colunas_ord = [c for c in ["f1_macro", "accuracy"] if c in df.columns]
    if not colunas_ord:
        return df.index.tolist()[:n]
    return df.sort_values(colunas_ord, ascending=False).head(n).index.tolist()


def _resolver_dispositivo_torch(dispositivo: str | None = None):
    import torch

    if dispositivo is not None:
        return torch.device(dispositivo)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obter_probabilidades_bilstm(
    modelo,
    textos: Iterable[str],
    vocabulario: dict[str, int],
    dataset_inferencia_cls,
    dataloader_cls,
    comprimento_maximo: int = 500,
    batch_size: int = 64,
    dispositivo: str | None = None,
) -> np.ndarray:
    """
    Retorna probabilidades (softmax) da BiLSTM para uma lista de textos.
    dataset_inferencia_cls e dataloader_cls sao injetados do notebook.
    """
    import torch

    dispositivo_torch = _resolver_dispositivo_torch(dispositivo)
    dataset = dataset_inferencia_cls(list(textos), vocabulario, comprimento_maximo)
    loader = dataloader_cls(dataset, batch_size=batch_size, shuffle=False)

    modelo = modelo.to(dispositivo_torch)
    modelo.eval()

    probabilidades = []
    with torch.no_grad():
        for lote_x in loader:
            logits = modelo(lote_x.to(dispositivo_torch))
            probabilidades.extend(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(probabilidades)


def obter_probabilidades_legal_bert(
    modelo,
    tokenizer,
    textos: Iterable[str],
    dataset_bert_cls,
    dataloader_cls,
    max_len: int = 512,
    batch_size: int = 32,
    dispositivo: str | None = None,
) -> np.ndarray:
    """
    Retorna probabilidades (softmax) do Legal-BERT para uma lista de textos.
    dataset_bert_cls e dataloader_cls sao injetados do notebook.
    """
    import torch

    dispositivo_torch = _resolver_dispositivo_torch(dispositivo)
    dataset = dataset_bert_cls(list(textos), rotulos=None, tokenizer=tokenizer, max_len=max_len)
    loader = dataloader_cls(dataset, batch_size=batch_size, shuffle=False)

    modelo = modelo.to(dispositivo_torch)
    modelo.eval()

    probabilidades = []
    with torch.no_grad():
        for lote in loader:
            b_ids = lote["input_ids"].to(dispositivo_torch)
            b_mask = lote["attention_mask"].to(dispositivo_torch)
            logits = modelo(input_ids=b_ids, attention_mask=b_mask).logits
            probabilidades.extend(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(probabilidades)


def combinar_probabilidades(
    probabilidades_a,
    probabilidades_b,
    peso_a: float = 0.5,
) -> np.ndarray:
    """
    Combina duas matrizes de probabilidade via media ponderada.
    """
    probs_a = np.asarray(probabilidades_a)
    probs_b = np.asarray(probabilidades_b)

    if probs_a.shape != probs_b.shape:
        raise ValueError(
            "As matrizes de probabilidade precisam ter o mesmo shape: "
            f"{probs_a.shape} vs {probs_b.shape}."
        )
    if not (0.0 <= peso_a <= 1.0):
        raise ValueError("peso_a deve estar entre 0 e 1.")

    return (probs_a * peso_a) + (probs_b * (1.0 - peso_a))


def predizer_por_probabilidades(probabilidades) -> np.ndarray:
    """Converte matriz de probabilidades em classes via argmax."""
    probs = np.asarray(probabilidades)
    if probs.ndim != 2:
        raise ValueError("As probabilidades devem estar no formato 2D [n_amostras, n_classes].")
    return np.argmax(probs, axis=1)


def gerar_submissao(
    ids,
    predicoes,
    nome_arquivo: str = "submission.csv",
) -> pd.DataFrame:
    """
    Gera arquivo CSV no formato Kaggle: colunas Id e Category.
    """
    ids_arr = np.asarray(ids)
    preds_arr = np.asarray(predicoes).astype(int)

    if ids_arr.shape[0] != preds_arr.shape[0]:
        raise ValueError(
            "Quantidade de IDs e predicoes deve ser igual: "
            f"ids={ids_arr.shape[0]} vs preds={preds_arr.shape[0]}"
        )

    df_sub = pd.DataFrame({"Id": ids_arr, "Category": preds_arr})
    df_sub.to_csv(nome_arquivo, index=False)
    print(f"Submissao salva em '{nome_arquivo}' com {len(df_sub):,} linhas.")
    return df_sub
