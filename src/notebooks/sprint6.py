"""
Planapp Sprint 6 — Avaliação Comparativa de Modelos de Classificação
=====================================================================
Objetivo:
  Classificar as 240 notícias do golden set com dois métodos:
    1. TF-IDF (baseline)
    2. LLM via Groq API (Llama 1B, expansível para 3B, 8B, etc.)

  Produzir um CSV com:
    id | titulo | Categoria_humana | cat_tfidf | cat_llm_1b | ...

  E calcular métricas de avaliação (accuracy, F1) por modelo.

Requisitos:
  pip install groq pandas scikit-learn nltk tqdm

Configuração:
  Exportar a chave da API Groq como variável de ambiente:
    export GROQ_API_KEY="gsk_..."
  Ou colocar diretamente em GROQ_API_KEY abaixo.
"""

import os
import time
import json
import re
import numpy as np
import pandas as pd
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from nltk.corpus import stopwords
from tqdm import tqdm
from groq import Groq

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Chave Groq — usa variável de ambiente (recomendado) ou coloca aqui
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "COLOCA_AQUI_A_TUA_CHAVE")

# Modelos a avaliar — adiciona mais à medida que queres comparar
LLM_MODELS = {
    "llm_8b": "llama-3.1-8b-instant",
    # "llm_70b": "llama-3.3-70b-versatile",  # descomenta para comparar
}

# Delay entre chamadas à API (segundos) — evita rate limit no tier gratuito
API_DELAY = 0.5

# Número máximo de tokens do corpo da notícia enviado ao LLM
# (para não exceder o context window e poupar tokens)
MAX_BODY_CHARS = 1500

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

def find_project_root(start: Path, marker: str = "data") -> Path:
    current = start
    for _ in range(6):
        if (current / marker).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Pasta '{marker}' não encontrada a partir de {start}")

PROJECT_ROOT = find_project_root(BASE_DIR)

GOLD_PATH   = PROJECT_ROOT / "data" / "gold" / "golden_set_10_por_categoria.parquet"
OUT_DIR     = PROJECT_ROOT / "data" / "evaluation"
OUT_CSV     = OUT_DIR / "avaliacao_modelos.csv"
OUT_METRICS = OUT_DIR / "metricas_modelos.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TAXONOMIA — lista oficial de categorias
# =============================================================================
CATEGORIAS = [
    "agricultura e floresta",
    "administração publica",
    "ambiente e clima",
    "cultura",
    "defesa",
    "desporto",
    "economia",
    "educação e formação",
    "energia",
    "habitação",
    "impostos",
    "i&d e inovação",
    "justiça",
    "mar e pescas",
    "proteção social",
    "saúde",
    "segurança",
    "trabalho",
    "transportes",
    "demografia",
    "desigualdades",
    "infraestruturas",
    "relações internacionais",
    "território",
]

CATEGORIAS_STR = "\n".join(f"- {c}" for c in CATEGORIAS)

# =============================================================================
# HELPERS
# =============================================================================

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalizar_categoria(texto: str) -> str:
    """
    Tenta mapear a resposta livre do LLM a uma das categorias oficiais.
    Faz matching por substring case-insensitive.
    """
    texto = texto.lower().strip()
    # match direto
    for cat in CATEGORIAS:
        if cat in texto:
            return cat
    # match parcial por palavras-chave principais
    mapa_parcial = {
        "agricult":       "agricultura e floresta",
        "floresta":       "agricultura e floresta",
        "incêndio":       "agricultura e floresta",
        "administraç":    "administração publica",
        "função pública": "administração publica",
        "ambiente":       "ambiente e clima",
        "clima":          "ambiente e clima",
        "emissõ":         "ambiente e clima",
        "cultura":        "cultura",
        "arte":           "cultura",
        "defesa":         "defesa",
        "militar":        "defesa",
        "nato":           "defesa",
        "desporto":       "desporto",
        "futebol":        "desporto",
        "economia":       "economia",
        "pib":            "economia",
        "mercado":        "economia",
        "educaç":         "educação e formação",
        "escola":         "educação e formação",
        "ensino":         "educação e formação",
        "energia":        "energia",
        "solar":          "energia",
        "eólica":         "energia",
        "habitaç":        "habitação",
        "arrendamento":   "habitação",
        "imobiliário":    "habitação",
        "imposto":        "impostos",
        "fiscal":         "impostos",
        "irs":            "impostos",
        "inovaç":         "i&d e inovação",
        "investigaç":     "i&d e inovação",
        "startup":        "i&d e inovação",
        "justiça":        "justiça",
        "tribunal":       "justiça",
        "crime":          "justiça",
        "pesca":          "mar e pescas",
        "pescador":       "mar e pescas",
        "mar":            "mar e pescas",
        "proteção social":"proteção social",
        "pensão":         "proteção social",
        "subsídio":       "proteção social",
        "saúde":          "saúde",
        "hospital":       "saúde",
        "médico":         "saúde",
        "segurança":      "segurança",
        "polícia":        "segurança",
        "trabalho":       "trabalho",
        "emprego":        "trabalho",
        "salário":        "trabalho",
        "transport":      "transportes",
        "mobilidade":     "transportes",
        "demograf":       "demografia",
        "populaç":        "demografia",
        "imigr":          "demografia",
        "desigual":       "desigualdades",
        "pobreza":        "desigualdades",
        "infraestrut":    "infraestruturas",
        "estrada":        "infraestruturas",
        "obra":           "infraestruturas",
        "relações intern":"relações internacionais",
        "diplomacia":     "relações internacionais",
        "território":     "território",
        "regional":       "território",
    }
    for kw, cat in mapa_parcial.items():
        if kw in texto:
            return cat
    return "desconhecido"


# =============================================================================
# MÉTODO 1 — TF-IDF BASELINE
# =============================================================================

def classificar_tfidf(df_gold: pd.DataFrame, stop_words: list) -> list[str]:
    """
    Cria um perfil TF-IDF por categoria (concatenando os corpos das notícias)
    e classifica cada notícia por similaridade de cosseno.
    """
    print("\n[TF-IDF] A construir perfis de categoria...")

    # Perfil de cada categoria = concatenação dos textos do golden set
    perfis = (
        df_gold.groupby("Categoria")["noticia_norm"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words=stop_words,
        ngram_range=(1, 2)
    )

    # Treinar nos perfis
    tfidf_perfis = vectorizer.fit_transform(perfis["noticia_norm"])

    # Transformar as notícias do golden set
    textos_gold = df_gold["noticia_norm"].astype(str).fillna("").tolist()
    tfidf_gold  = vectorizer.transform(textos_gold)

    # Similaridade cosseno
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(tfidf_gold, tfidf_perfis)

    # Categoria com maior score
    best_idx = sims.argmax(axis=1)
    preds = [perfis.iloc[i]["Categoria"] for i in best_idx]

    print(f"[TF-IDF] Classificação concluída ({len(preds)} notícias)")
    return preds


# =============================================================================
# MÉTODO 2 — LLM via GROQ
# =============================================================================

def construir_prompt(titulo: str, corpo: str) -> str:
    corpo_truncado = corpo[:MAX_BODY_CHARS]
    return f"""És um sistema de classificação de notícias portuguesas.

Classifica a notícia abaixo numa das seguintes categorias (responde APENAS com o nome exato da categoria, sem mais texto):

{CATEGORIAS_STR}

TÍTULO: {titulo}

CORPO: {corpo_truncado}

CATEGORIA:"""


def classificar_llm(
    df_gold: pd.DataFrame,
    model_id: str,
    model_label: str,
    client: Groq,
) -> list[str]:
    """
    Classifica cada notícia usando o LLM via Groq API.
    Guarda checkpoint a cada 50 notícias para não perder progresso.
    """
    checkpoint_path = OUT_DIR / f"checkpoint_{model_label}.json"

    # Retomar de checkpoint se existir
    resultados = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            resultados = json.load(f)
        print(f"[{model_label}] Checkpoint encontrado — {len(resultados)} notícias já classificadas")

    ids = df_gold["id"].astype(str).tolist()
    titulos = df_gold["titulo"].astype(str).tolist()
    corpos  = df_gold["noticia_norm"].astype(str).tolist()

    print(f"\n[{model_label}] A classificar {len(ids)} notícias com {model_id}...")

    for i, (nid, titulo, corpo) in enumerate(
        tqdm(zip(ids, titulos, corpos), total=len(ids), desc=model_label)
    ):
        if nid in resultados:
            continue  # já classificado

        prompt = construir_prompt(titulo, corpo)

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0,
            )
            resposta_raw = response.choices[0].message.content.strip()
            categoria = normalizar_categoria(resposta_raw)
            resultados[nid] = {
                "raw": resposta_raw,
                "categoria": categoria,
            }
        except Exception as e:
            print(f"\n⚠️  Erro na notícia {nid}: {e}")
            resultados[nid] = {"raw": "ERRO", "categoria": "desconhecido"}

        # Checkpoint a cada 50
        if (i + 1) % 50 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
            print(f"\n  💾 Checkpoint guardado ({i+1}/{len(ids)})")

        time.sleep(API_DELAY)

    # Guardar checkpoint final
    with open(checkpoint_path, "w") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Devolver na ordem do dataframe
    preds = [resultados.get(nid, {}).get("categoria", "desconhecido") for nid in ids]
    print(f"[{model_label}] Concluído ✅")
    return preds


# =============================================================================
# MÉTRICAS
# =============================================================================

def calcular_metricas(y_true: list, y_pred: list, nome_modelo: str) -> str:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, zero_division=0)

    linhas = [
        "=" * 70,
        f"MODELO: {nome_modelo}",
        "=" * 70,
        f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)",
        f"  F1 macro:    {f1_macro:.4f}",
        f"  F1 weighted: {f1_weighted:.4f}",
        "",
        "Relatório por categoria:",
        report,
    ]
    return "\n".join(linhas)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Verificar API key ──────────────────────────────────────────────────
    if GROQ_API_KEY == "COLOCA_AQUI_A_TUA_CHAVE":
        raise ValueError(
            "❌ Chave Groq não configurada!\n"
            "   Exporta como variável de ambiente: export GROQ_API_KEY='gsk_...'\n"
            "   Ou coloca diretamente na variável GROQ_API_KEY no topo do script."
        )

    # ── Stopwords ─────────────────────────────────────────────────────────
    try:
        stop_words_pt = list(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords")
        stop_words_pt = list(stopwords.words("portuguese"))

    # ── Carregar golden set ────────────────────────────────────────────────
    print(f"A carregar golden set: {GOLD_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Golden set não encontrado em {GOLD_PATH}\n"
            "Corre primeiro o script 10noticias_categoria.py"
        )

    df_gold = pd.read_parquet(GOLD_PATH)
    print(f"Golden set carregado: {df_gold.shape}")
    print(f"Colunas: {list(df_gold.columns)}")

    # Garantir coluna id como string para o checkpoint
    df_gold["id"] = df_gold["id"].astype(str)

    y_true = df_gold["Categoria"].tolist()
    print(f"\nDistribuição do human label:")
    print(df_gold["Categoria"].value_counts().to_string())

    # ── Método 1: TF-IDF ──────────────────────────────────────────────────
    preds_tfidf = classificar_tfidf(df_gold, stop_words_pt)

    # ── Método 2: LLM via Groq ────────────────────────────────────────────
    client = Groq(api_key=GROQ_API_KEY)

    preds_llm = {}
    for model_label, model_id in LLM_MODELS.items():
        preds_llm[model_label] = classificar_llm(
            df_gold, model_id, model_label, client
        )

    # ── Construir CSV de avaliação ─────────────────────────────────────────
    df_out = df_gold[["id", "titulo", "Categoria"]].copy()
    df_out = df_out.rename(columns={"Categoria": "Categoria_humana"})
    df_out["cat_tfidf"] = preds_tfidf

    for model_label, preds in preds_llm.items():
        df_out[f"cat_{model_label}"] = preds

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV de avaliação guardado em: {OUT_CSV}")
    print(df_out.head(10).to_string(index=False))

    # ── Métricas ──────────────────────────────────────────────────────────
    all_metrics = []

    metricas_tfidf = calcular_metricas(y_true, preds_tfidf, "TF-IDF (baseline)")
    all_metrics.append(metricas_tfidf)
    print("\n" + metricas_tfidf)

    for model_label, preds in preds_llm.items():
        m = calcular_metricas(y_true, preds, model_label.upper())
        all_metrics.append(m)
        print("\n" + m)

    # ── Resumo comparativo ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESUMO COMPARATIVO")
    print("=" * 70)

    modelos = ["TF-IDF"] + list(LLM_MODELS.keys())
    todas_preds = [preds_tfidf] + list(preds_llm.values())

    for nome, preds in zip(modelos, todas_preds):
        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds, average="macro", zero_division=0)
        print(f"  {nome:<20} Accuracy: {acc*100:.1f}%   F1-macro: {f1:.4f}")

    # Guardar métricas em ficheiro
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_metrics))
    print(f"\n📄 Métricas detalhadas guardadas em: {OUT_METRICS}")


if __name__ == "__main__":
    main()