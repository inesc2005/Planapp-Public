import os
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "cision_news_20251110.parquet")
GOLD_PATH = os.path.join(PROJECT_ROOT, "data_sample", "golden_set_10_por_categoria.parquet")

OUT_DIR = os.path.join(PROJECT_ROOT, "data_sample")
OUT_PARQUET = os.path.join(OUT_DIR, "silver_com_categoria_vfinal.parquet")
OUT_CSV = os.path.join(OUT_DIR, "silver_com_categoria_vfinal.csv")

# =========================
# CONFIG
# =========================
ID_COL = "id"
TEXT_COL = "noticia_norm"
TITLE_COL = "titulo"
GOLD_CAT_COL = "Categoria"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
TOP_K = 5  # Para o KNN de Embeddings

# =========================
# FUNÇÕES
# =========================

def ensure_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Faltam colunas: {missing}\nDisponíveis: {list(df.columns)}")

def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def predict_with_knn(emb_x: np.ndarray, emb_gold, gold_ids, gold_titles, gold_cats, top_k=5):
    """Método 1: Similaridade por Embeddings (Semântica)"""
    sims = cosine_similarity(emb_x.reshape(1, -1), emb_gold).ravel()
    idx = np.argpartition(-sims, kth=min(top_k, len(sims)-1))[:top_k]
    idx = idx[np.argsort(-sims[idx])]

    top1 = idx[0]
    sim_best = float(sims[top1])
    
    score = defaultdict(float)
    for j in idx:
        score[str(gold_cats[j])] += float(sims[j])

    pred_cat = max(score.items(), key=lambda x: x[1])[0]
    return pred_cat, sim_best

def predict_with_tfidf_profiles(df_gold, silver_texts, stop_words):
    """
    Cria um 'Vetor Mestre' para cada categoria usando o CORPO das notícias (noticia_norm) 
    e compara com a base Silver.
    """
    # 1. Agrupar o CORPO das notícias do Golden Set por Categoria
    # Usamos TEXT_COL ("noticia_norm") em vez de TITLE_COL
    gold_profiles = df_gold.groupby(GOLD_CAT_COL)[TEXT_COL].apply(lambda x: " ".join(x.astype(str))).reset_index()
    
    # 2. Treinar o Vectorizer nos perfis (corpo completo)
    # Aumentamos para 5000 features para capturar mais detalhes do corpo
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1,2))
    tfidf_gold = vectorizer.fit_transform(gold_profiles[TEXT_COL])
    
    # 3. Transformar o corpo das notícias Silver
    tfidf_silver = vectorizer.transform(silver_texts)
    
    # 4. Calcular similaridade de cosseno
    sims = cosine_similarity(tfidf_silver, tfidf_gold)
    
    # 5. Extrair a categoria com maior score
    best_idx = sims.argmax(axis=1)
    preds_cat = [gold_profiles.iloc[i][GOLD_CAT_COL] for i in best_idx]
    preds_score = sims.max(axis=1)
    
    return preds_cat, preds_score

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Preparar Stopwords
    try:
        stop_words_pt = list(stopwords.words("portuguese"))
    except:
        nltk.download('stopwords')
        stop_words_pt = list(stopwords.words("portuguese"))

    # Carregar dados
    df_silver = pd.read_parquet(SILVER_PATH)
    df_gold = pd.read_parquet(GOLD_PATH)

    ensure_columns(df_silver, [ID_COL, TITLE_COL, TEXT_COL], "SILVER")
    ensure_columns(df_gold, [ID_COL, TITLE_COL, TEXT_COL, GOLD_CAT_COL], "GOLDEN")

    silver_texts = df_silver[TEXT_COL].astype(str).fillna("").tolist()
    gold_texts = df_gold[TEXT_COL].astype(str).fillna("").tolist()

    print(f"Golden: {len(df_gold)} | Silver: {len(df_silver)}")

    # --- EXECUÇÃO MÉTODO 1: EMBEDDINGS ---
    print("\n[1/2] A calcular classificação via Embeddings (MiniLM)...")
    model = SentenceTransformer(MODEL_NAME)
    emb_gold = encode_texts(model, gold_texts, batch_size=BATCH_SIZE)
    emb_silver = encode_texts(model, silver_texts, batch_size=BATCH_SIZE)

    gold_cats = df_gold[GOLD_CAT_COL].astype(str).to_numpy()
    gold_ids = df_gold[ID_COL].to_numpy()
    gold_titles = df_gold[TITLE_COL].to_numpy()

    knn_preds = []
    knn_sims = []
    for i in tqdm(range(len(df_silver)), desc="KNN"):
        cat, sim = predict_with_knn(emb_silver[i], emb_gold, gold_ids, gold_titles, gold_cats, top_k=TOP_K)
        knn_preds.append(cat)
        knn_sims.append(sim)

    # --- EXECUÇÃO MÉTODO 2: TF-IDF (O pedido pelo professor) ---
    print("\n[2/2] A calcular classificação via TF-IDF (Perfis de Categoria)...")
    tfidf_preds, tfidf_scores = predict_with_tfidf_profiles(df_gold, silver_texts, stop_words_pt)

    # --- COMPILAÇÃO DE RESULTADOS ---
    df_out = df_silver.copy()
    df_out["cat_embeddings"] = knn_preds
    df_out["score_embeddings"] = knn_sims
    df_out["cat_tfidf"] = tfidf_preds
    df_out["score_tfidf"] = tfidf_scores

    # Guardar resultados
    df_out.to_parquet(OUT_PARQUET, index=False)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n✅ Classificação terminada! Guardado em: {OUT_CSV}")

    # PRINT DE EVIDÊNCIA PARA O PROFESSOR
    print("\n" + "="*80)
    print("AMOSTRA DE RESULTADOS (Comparação de Métodos)")
    print("="*80)
    cols_show = [TITLE_COL, "cat_embeddings", "cat_tfidf", "score_tfidf"]
    print(df_out[cols_show].head(10).to_string(index=False))

if __name__ == "__main__":
    main()