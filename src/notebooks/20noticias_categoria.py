import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIG
# =========================
ID_COL = "id"
TITLE_COL = "titulo"
TEXT_COL = "noticia_norm"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


# =========================
# TAXONOMIA
# =========================
CATEGORIES = {
    "Agricultura e Floresta": "agricultura florestas incêndios gestão florestal recursos rurais",
    "Administração Pública": "administração pública serviços públicos estado autarquias burocracia",
    "Ambiente e Clima": "ambiente clima emissões sustentabilidade poluição biodiversidade alterações climáticas conservação",
    "Cultura": "cultura artes património museus literatura cinema música exposições festivais",
    "Defesa": "defesa forças armadas estratégia militar nato missões armamento",
    "Desporto": "desporto futebol jogos atletas campeonato clube treinador liga taça modalidade",
    "Economia": "economia empresas mercado investimento inflação pib banca juros exportações",
    "Educação e Formação": "educação escolas universidades ensino professores alunos formação exames currículo",
    "Energia": "energia electricidade renováveis gás solar eólica hidrogénio combustíveis",
    "Habitação": "habitação imobiliário arrendamento casas rendas construção alojamento",
    "Impostos e taxas": "impostos fiscalidade irs iva taxas tributação finanças",
    "I&D e Inovação": "investigação inovação ciência tecnologia startups laboratório patentes",
    "Justiça": "justiça tribunais processos juiz crime ministério público arguido acórdão",
    "Mar e Pescas": "mar pescas pescadores aquacultura oceano recursos marinhos quotas",
    "Proteção Social": "proteção social segurança social pensões apoios subsídios prestações inclusão",
    "Saúde": "saúde hospitais sns médicos vacinação medicamentos consultas urgência",
    "Segurança": "segurança polícia crime gnr psp emergência proteção civil detenção",
    "Trabalho": "trabalho emprego salários sindicatos greve contrato desemprego precariedade trabalhadores",
    "Transportes e Mobilidade": "transportes mobilidade metro comboio cp aeroporto autocarro trânsito ferrovia rodovia",
    "Demografia": "demografia população natalidade envelhecimento migração imigração residentes censos",
    "Desigualdade": "desigualdade pobreza exclusão social rendimento vulnerabilidade disparidades",
    "Infraestruturas": "infraestruturas estradas pontes obras públicas saneamento rede construção",
    "Relações Internacionais": "relações internacionais diplomacia união europeia nato onu política externa geopolítica",
    "Território": "território ordenamento municípios regiões coesão territorial planeamento regionalização interior litoral",
}


# =========================
# HELPERS
# =========================
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def build_hybrid_text(title: str, body: str) -> str:
    """
    Junta título + corpo para melhorar embeddings.
    """
    title = safe_str(title).strip()
    body = safe_str(body).strip()

    if not body:
        return title

    if len(body) < 80:
        return f"TITULO: {title}\nCORPO: {body}\n{title} {title}"

    return f"TITULO: {title}\nCORPO: {body}\n{title}"


def detect_default_silver_path() -> str:
    """
    Tenta encontrar automaticamente o silver a partir da localização do script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return os.path.join(project_root, "data", "silver", "cision_news_20251110.parquet")


def ensure_required_columns(df: pd.DataFrame):
    required = [ID_COL, TITLE_COL, TEXT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltam colunas obrigatórias: {missing}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )


def print_preview(df_final: pd.DataFrame, per_cat: int):
    print("\n" + "=" * 120)
    print("PREVIEW — TOP notícias por categoria")
    print("=" * 120)

    for cat in df_final["categoria"].unique():
        sub = df_final[df_final["categoria"] == cat].copy()
        sub = sub.sort_values("similarity", ascending=False)

        print("\n" + "-" * 120)
        print(f"CATEGORIA: {cat} | N={len(sub)}")
        print("-" * 120)

        for _, row in sub.head(per_cat).iterrows():
            nid = row[ID_COL]
            sim = row["similarity"]
            titulo = safe_str(row[TITLE_COL])[:140]
            corpo = safe_str(row[TEXT_COL]).replace("\n", " ")
            corpo = corpo[:220] + ("..." if len(corpo) > 220 else "")

            print(f"- id={nid} | sim={sim:.4f} | {titulo}")
            print(f"  corpo: {corpo}")


# =========================
# MAIN
# =========================
def main():
    default_silver = detect_default_silver_path()

    parser = argparse.ArgumentParser(
        description="Gerar 20 notícias por categoria a partir do silver."
    )
    parser.add_argument(
        "--silver_path",
        default=default_silver,
        help="Caminho para o parquet silver"
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_sample"),
        help="Pasta de output"
    )
    parser.add_argument(
        "--per_cat",
        type=int,
        default=20,
        help="Número de notícias por categoria"
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=0,
        help="Se > 0, usa apenas uma amostra aleatória de N notícias"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mostra preview no terminal"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("A carregar dataset...")
    print(f"Silver path: {args.silver_path}")

    if not os.path.exists(args.silver_path):
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {args.silver_path}"
        )

    df = pd.read_parquet(args.silver_path)
    ensure_required_columns(df)

    print(f"Dataset carregado com {len(df)} linhas.")

    # limpar nulos
    df = df.copy()
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    # amostra opcional
    if args.sample_n and args.sample_n > 0:
        sample_size = min(args.sample_n, len(df))
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"Amostra aleatória usada: {len(df)} notícias.")

    # texto híbrido
    df["hybrid_text"] = [
        build_hybrid_text(t, b)
        for t, b in zip(df[TITLE_COL], df[TEXT_COL])
    ]

    # remover textos vazios
    df = df[df["hybrid_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"Após limpeza: {len(df)} notícias.")

    print("A carregar modelo de embeddings...")
    model = SentenceTransformer(MODEL_NAME)

    print("A gerar embeddings das notícias...")
    news_embeddings = model.encode(
        df["hybrid_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("A gerar embeddings das categorias...")
    category_names = list(CATEGORIES.keys())
    category_texts = list(CATEGORIES.values())

    category_embeddings = model.encode(
        category_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("A calcular similaridades...")
    sims = cosine_similarity(news_embeddings, category_embeddings)

    all_results = []

    for i, cat in enumerate(category_names):
        df_cat = df[[ID_COL, TITLE_COL, TEXT_COL]].copy()
        df_cat["categoria"] = cat
        df_cat["similarity"] = sims[:, i]

        df_cat = df_cat.sort_values("similarity", ascending=False).head(args.per_cat)
        all_results.append(df_cat)

    final = pd.concat(all_results, axis=0).reset_index(drop=True)

    # outputs
    output_csv = os.path.join(args.out_dir, f"golden_candidates_{args.per_cat}_por_categoria.csv")
    output_parquet = os.path.join(args.out_dir, f"golden_candidates_{args.per_cat}_por_categoria.parquet")

    final.to_csv(output_csv, index=False, encoding="utf-8")
    final.to_parquet(output_parquet, index=False)

    print("\n✅ Dataset gerado com sucesso")
    print(f"CSV: {output_csv}")
    print(f"Parquet: {output_parquet}")

    print("\nNotícias por categoria:")
    print(final["categoria"].value_counts().to_string())

    if args.preview:
        print_preview(final, args.per_cat)


if __name__ == "__main__":
    main()