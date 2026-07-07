# -*- coding: utf-8 -*-
"""
Bronze -> Silver (líderes) escalável
- Lê o Bronze mais recente em data/bronze (cision_news_YYYYMMDD.{csv,parquet})
- Limpa texto (noticia_norm) e vetoriza TF-IDF (1–2 grams)
- Constrói grafo kNN (cosine) e aplica threshold; grupos = componentes conexos
- Líder = maior valor_publicitario em cada grupo
- Exporta para data/silver/{<prefixo_bronze>}{YYYYMMDD}.{csv,parquet}
"""

import os, re, time, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

# Defaults (ajustáveis por CLI)
THRESHOLD = 0.76
K_NEIGHBORS = 25
MAX_FEATURES = 30000
MIN_DF = 3
NGRAMS = (1, 2)

BRONZE_DIR = os.path.join("data", "bronze")
SILVER_DIR = os.path.join("data", "silver")

# ---------------- Utils ----------------
def norm_txt(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[\:\;\,\.\!\?\(\)\[\]\"\'\-/]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def find_latest_bronze():
    if not os.path.isdir(BRONZE_DIR):
        return None, None
    files = sorted([f for f in os.listdir(BRONZE_DIR) if f.startswith("cision_news_")], reverse=True)
    parquet = next((os.path.join(BRONZE_DIR, f) for f in files if f.endswith(".parquet")), None)
    csv = next((os.path.join(BRONZE_DIR, f) for f in files if f.endswith(".csv")), None)
    return parquet, csv

def bronze_prefix():
    pq, csv = find_latest_bronze()
    src = pq or csv
    if src:
        base = os.path.basename(src)
        m = re.match(r"(cision_news_)\\d{8}\\.(csv|parquet)$", base)
        if m:
            return m.group(1)
    return "cision_news_"

def load_bronze():
    pq, csv = find_latest_bronze()
    if pq and os.path.exists(pq):
        df = pd.read_parquet(pq, engine="pyarrow")
    elif csv and os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Bronze não encontrado em data/bronze.")

    # Tipos e texto
    df["data_publicacao"] = pd.to_datetime(df.get("data_publicacao"), errors="coerce")
    df["valor_publicitario"] = pd.to_numeric(df.get("valor_publicitario"), errors="coerce")
    df["noticia_raw"] = df["noticia"].fillna("").astype(str)
    df["noticia_norm"] = df["noticia_raw"].map(norm_txt)
    return df

def build_tfidf(texts: pd.Series):
    vec = TfidfVectorizer(ngram_range=NGRAMS, max_features=MAX_FEATURES, min_df=MIN_DF)
    return vec.fit_transform(texts)

def group_by_knn_threshold(X, k=K_NEIGHBORS, threshold=THRESHOLD):
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1).fit(X)
    G = nbrs.kneighbors_graph(mode="distance").tocsr()  # dist = 1 - cos
    G.data = 1.0 - G.data
    G.data[G.data < threshold] = 0.0
    G.eliminate_zeros()
    B = G.copy(); B.data[:] = 1.0
    n_comp, labels = connected_components(csgraph=B, directed=False, return_labels=True)

    grupos = {}
    for i, g in enumerate(labels):
        grupos.setdefault(g, []).append(i)
    grupos = list(grupos.values())
    gid_map = {idx: gid for gid, g in enumerate(grupos) for idx in g}
    return grupos, gid_map

def pick_leaders(df: pd.DataFrame, grupos: list[list[int]], col="valor_publicitario") -> list[int]:
    leaders = []
    for g in grupos:
        if not g: continue
        valid = [i for i in g if 0 <= int(i) < len(df)]
        if not valid: continue
        sub = df.loc[valid, col]
        lid = int(sub.idxmax()) if sub.notna().any() else int(valid[0])
        leaders.append(lid)
    return leaders

def export_silver(silver_df: pd.DataFrame):
    os.makedirs(SILVER_DIR, exist_ok=True)
    datestamp = datetime.utcnow().strftime("%Y%m%d")
    prefix = bronze_prefix()  # ex.: "cision_news_"
    csv_path = os.path.join(SILVER_DIR, f"{prefix}{datestamp}.csv")
    pq_path  = os.path.join(SILVER_DIR, f"{prefix}{datestamp}.parquet")

    cols_pref = [
        "id","titulo","noticia_raw","noticia_norm","link",
        "data_publicacao","tipo_meio","autores",
        "valor_publicitario","guid","categoria",
        "grupo_id","is_lider","processed_at"
    ]
    cols = [c for c in cols_pref if c in silver_df.columns]
    silver_df[cols].to_csv(csv_path, index=False, encoding="utf-8")
    try:
        silver_df[cols].to_parquet(pq_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"[Aviso] Parquet falhou ({e}); CSV foi gravado.")
    print(f"[OK] Silver gravado:\n - {csv_path}\n - {pq_path}")

# ---------------- Main ----------------
def main(args):
    t0 = time.time()
    print("[1/5] Carregar Bronze...")
    df = load_bronze()
    print(f"Registos: {len(df)}")

    print("[2/5] TF-IDF...")
    X = build_tfidf(df["noticia_norm"])
    print(f"TF-IDF shape: {X.shape}")

    print(f"[3/5] kNN (k={args.k}) + threshold ({args.threshold}) -> grupos...")
    grupos, gid_map = group_by_knn_threshold(X, k=args.k, threshold=args.threshold)
    df = df.reset_index(drop=True)
    df["grupo_id"] = df.index.map(gid_map.get).astype("Int64")
    print(f"Grupos: {len(grupos)}")

    print("[4/5] Líder por valor_publicitario...")
    leaders_idx = pick_leaders(df, grupos, col="valor_publicitario")
    df["is_lider"] = False
    df.loc[leaders_idx, "is_lider"] = True
    df["processed_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    silver = df.loc[df["is_lider"]].copy()
    print(f"Líderes: {len(silver)}")

    print("[5/5] Exportar Silver...")
    export_silver(silver)
    print(f"[DONE] {time.time()-t0:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument("--max_features", type=int, default=MAX_FEATURES)
    parser.add_argument("--min_df", type=int, default=MIN_DF)
    args = parser.parse_args()

    # aplicar overrides globais de TF-IDF
    THRESHOLD = args.threshold
    K_NEIGHBORS = args.k
    MAX_FEATURES = args.max_features
    MIN_DF = args.min_df

    main(args)
