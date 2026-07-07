"""
Avaliação Comparativa de Modelos
=================================
Carrega os JSONs de predições (TF-IDF e LLM) e o golden set,
produz o CSV final de avaliação e as métricas por modelo.

Inputs esperados em data/evaluation/:
  preds_tfidf.json
  preds_llm_llm_8b.json   (e/ou outros modelos LLM)

Outputs:
  data/evaluation/avaliacao_final.csv
  data/evaluation/metricas_final.txt
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

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
GOLD_PATH    = PROJECT_ROOT / "data" / "gold" / "golden_set_10_por_categoria.parquet"
EVAL_DIR     = PROJECT_ROOT / "data" / "evaluation"
OUT_CSV      = EVAL_DIR / "avaliacao_final.csv"
OUT_METRICS  = EVAL_DIR / "metricas_final.txt"

# =============================================================================
# HELPERS
# =============================================================================

def carregar_preds(path: Path, id_col: str = "categoria") -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {nid: v[id_col] for nid, v in data.items()}


def calcular_metricas(y_true: list, y_pred: list, nome: str) -> str:
    acc        = accuracy_score(y_true, y_pred)
    f1_macro   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report     = classification_report(y_true, y_pred, zero_division=0)

    return "\n".join([
        "=" * 70,
        f"MODELO: {nome}",
        "=" * 70,
        f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)",
        f"  F1 macro:    {f1_macro:.4f}",
        f"  F1 weighted: {f1_weighted:.4f}",
        "",
        "Relatorio por categoria:",
        report,
    ])


def resumo_probs(preds_json_path: Path, df_gold: pd.DataFrame, modelo_label: str):
    """
    Imprime, para cada notícia mal classificada, as top-3 probabilidades
    para ajudar a perceber onde o modelo hesita.
    """
    with open(preds_json_path, encoding="utf-8") as f:
        data = json.load(f)

    erros = []
    for _, row in df_gold.iterrows():
        nid = str(row["id"])
        if nid not in data:
            continue
        pred = data[nid]["categoria"]
        true = row["Categoria"]
        if pred != true:
            top3 = sorted(data[nid]["probs"].items(), key=lambda x: x[1], reverse=True)[:3]
            erros.append({
                "id":     nid,
                "titulo": row["titulo"][:50],
                "real":   true,
                "pred":   pred,
                "top3":   top3,
            })

    print(f"\n--- {modelo_label}: {len(erros)} erros ---")
    for e in erros[:10]:  # mostra só os primeiros 10
        print(f"  [{e['real']} -> {e['pred']}] {e['titulo']}...")
        print(f"    Top-3: {e['top3']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"A carregar golden set: {GOLD_PATH}")
    df_gold = pd.read_parquet(GOLD_PATH)
    df_gold["id"] = df_gold["id"].astype(str)
    y_true = df_gold["Categoria"].tolist()
    ids    = df_gold["id"].tolist()

    # ── Descobrir ficheiros de predições disponíveis ──────────────────────
    modelos = {}

    tfidf_path = EVAL_DIR / "preds_tfidf.json"
    if tfidf_path.exists():
        modelos["TF-IDF (descrições)"] = tfidf_path
    else:
        print(f"AVISO: {tfidf_path} não encontrado — corre primeiro tfidf_descricoes.py")

    emb_path = EVAL_DIR / "preds_embeddings.json"
    if emb_path.exists():
        modelos["Embeddings Semânticos"] = emb_path

    emb_e5_path = EVAL_DIR / "preds_embeddings_e5.json"
    if emb_e5_path.exists():
        modelos["Embeddings E5-Large"] = emb_e5_path

    bge_path = EVAL_DIR / "preds_embeddings_bge_m3.json"
    if bge_path.exists():
        modelos["Embeddings BGE-M3"] = bge_path

    for p in sorted(EVAL_DIR.glob("preds_llm_*.json")):
        label = p.stem.replace("preds_llm_", "LLM ").replace("_", " ").upper()
        modelos[label] = p

    if not modelos:
        raise FileNotFoundError(
            "Nenhum ficheiro de predições encontrado em data/evaluation/\n"
            "Corre primeiro tfidf_descricoes.py e llm_groq.py"
        )

    # ── Construir CSV de avaliação ─────────────────────────────────────────
    df_out = df_gold[["id", "titulo", "Categoria"]].copy()
    df_out = df_out.rename(columns={"Categoria": "Categoria_humana"})

    col_map = {
        "TF-IDF (descrições)": "cat_tfidf",
    }

    all_metrics = []
    resumo_linhas = []

    for nome_modelo, path in modelos.items():
        preds_dict = carregar_preds(path)
        preds = [preds_dict.get(nid, "desconhecido") for nid in ids]

        col = col_map.get(nome_modelo, f"cat_{path.stem.replace('preds_', '')}")
        df_out[col] = preds

        m = calcular_metricas(y_true, preds, nome_modelo)
        all_metrics.append(m)
        print("\n" + m)

        acc       = accuracy_score(y_true, preds)
        f1_macro  = f1_score(y_true, preds, average="macro", zero_division=0)
        precision = precision_score(y_true, preds, average="macro", zero_division=0)
        recall    = recall_score(y_true, preds, average="macro", zero_division=0)
        acertos   = sum(p == t for p, t in zip(preds, y_true))
        resumo_linhas.append((nome_modelo, acc, precision, recall, f1_macro, acertos, path))

    # ── Guardar CSV ────────────────────────────────────────────────────────
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCSV guardado em: {OUT_CSV}")
    print(df_out.head(5).to_string(index=False))

    # ── Guardar métricas ───────────────────────────────────────────────────
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_metrics))
    print(f"\nMetricas guardadas em: {OUT_METRICS}")

    # ── Tabela comparativa ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA DE MODELOS")
    print("=" * 80)
    header = f"  {'Modelo':<25} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1-Macro':>9} {'Acertos':>9}"
    print(header)
    print("-" * 80)
    resumo_linhas_sorted = sorted(resumo_linhas, key=lambda x: x[1], reverse=True)
    for nome, acc, prec, rec, f1, acertos, _ in resumo_linhas_sorted:
        print(f"  {nome:<25} {acc*100:>8.1f}% {prec*100:>8.1f}% {rec*100:>8.1f}% {f1*100:>8.1f}% {acertos:>6}/240")
    print("=" * 80)

    # ── Guardar tabela em CSV separado ────────────────────────────────────
    df_tabela = pd.DataFrame([
        {"Modelo": n, "Accuracy": f"{a*100:.1f}%", "Precision": f"{p*100:.1f}%",
         "Recall": f"{r*100:.1f}%", "F1-Macro": f"{f*100:.1f}%", "Acertos": f"{ac}/240"}
        for n, a, p, r, f, ac, _ in resumo_linhas_sorted
    ])
    tabela_path = EVAL_DIR / "tabela_comparativa.csv"
    df_tabela.to_csv(tabela_path, index=False, encoding="utf-8-sig")
    print(f"\nTabela guardada em: {tabela_path}")

    # ── Análise de erros com probabilidades ───────────────────────────────
    print("\n" + "=" * 80)
    print("ANALISE DE ERROS (top-3 probabilidades por noticia mal classificada)")
    print("=" * 80)
    for nome, acc, prec, rec, f1, acertos, path in resumo_linhas:
        resumo_probs(path, df_gold, nome)


if __name__ == "__main__":
    main()
