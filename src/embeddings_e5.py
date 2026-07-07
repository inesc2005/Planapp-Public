"""
Embeddings com multilingual-e5-large
=====================================
Classifica as 240 notícias do golden set usando o modelo
intfloat/multilingual-e5-large — maior dimensão (1024), mais contexto (512 tokens)
e melhor performance para português que o paraphrase-multilingual-mpnet-base-v2.

Nota: o e5 requer prefixo "query: " nas notícias e "passage: " nas descrições.

Output:
  data/evaluation/preds_embeddings_e5.json
    { id: { "categoria": str, "probs": {cat: float, ...} } }
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
OUT_DIR      = PROJECT_ROOT / "data" / "evaluation"
OUT_FILE     = OUT_DIR / "preds_embeddings_e5.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-large"

# =============================================================================
# DESCRIÇÕES EM LINGUAGEM NATURAL (optimizadas para embeddings — E5 requer frases)
# =============================================================================
DESCRICOES = {
    "agricultura e floresta": (
        "Notícias sobre agricultura, silvicultura e gestão florestal em Portugal, "
        "incluindo cultivo de vinhas e olivais, prevenção e combate a incêndios florestais, "
        "sapadores florestais, reflorestação, pecuária, regadio e cooperativas agrícolas."
    ),
    "administração publica": (
        "Notícias sobre o funcionamento da administração pública portuguesa, "
        "incluindo serviços do Estado, ministérios, municípios, câmaras municipais, "
        "funcionários públicos, concursos públicos, reformas administrativas e descentralização."
    ),
    "ambiente e clima": (
        "Notícias sobre alterações climáticas, sustentabilidade ambiental e proteção da natureza, "
        "incluindo emissões de carbono, energias renováveis, poluição, reciclagem, "
        "biodiversidade, cheias, secas e acordos climáticos internacionais."
    ),
    "cultura": (
        "Notícias sobre arte, cultura e entretenimento em Portugal, "
        "incluindo música, cinema, teatro, literatura, museus, festivais, "
        "património cultural, arqueologia, prémios literários e espetáculos."
    ),
    "defesa": (
        "Notícias sobre as forças armadas portuguesas e segurança nacional, "
        "incluindo exército, marinha, força aérea, NATO, missões militares internacionais, "
        "armamento, orçamento de defesa e cooperação militar europeia."
    ),
    "desporto": (
        "Notícias sobre desporto e competições desportivas em Portugal, "
        "incluindo futebol, atletismo, natação, ciclismo, seleção nacional, "
        "clubes, campeonatos, transferências de jogadores e resultados desportivos."
    ),
    "economia": (
        "Notícias sobre economia e finanças em Portugal e na Europa, "
        "incluindo crescimento económico, inflação, mercados financeiros, empresas, "
        "investimento, exportações, dívida pública, emprego e desemprego."
    ),
    "educação e formação": (
        "Notícias sobre o sistema de ensino e educação em Portugal, "
        "incluindo escolas, universidades, alunos, professores, exames, "
        "ensino superior, formação profissional, literacia e políticas educativas."
    ),
    "energia": (
        "Notícias sobre produção e consumo de energia em Portugal, "
        "incluindo energias renováveis como solar e eólica, eletricidade, gás natural, "
        "petróleo, tarifas energéticas, descarbonização e transição energética."
    ),
    "habitação": (
        "Notícias sobre habitação e mercado imobiliário em Portugal, "
        "incluindo arrendamento, preços de casas, habitação pública, crédito habitação, "
        "construção, reabilitação urbana, sem-abrigo e alojamento local."
    ),
    "impostos": (
        "Notícias sobre fiscalidade e impostos em Portugal, "
        "incluindo IRS, IRC, IVA, autoridade tributária, benefícios fiscais, "
        "fraude fiscal, evasão fiscal, deduções e reformas do sistema fiscal."
    ),
    "i&d e inovação": (
        "Notícias sobre investigação científica, inovação e tecnologia em Portugal, "
        "incluindo startups, inteligência artificial, digitalização, laboratórios, "
        "patentes, fundos europeus para investigação, robótica e biotecnologia."
    ),
    "justiça": (
        "Notícias sobre o sistema judicial e justiça em Portugal, "
        "incluindo tribunais, julgamentos, corrupção, processos criminais, "
        "magistrados, advogados, ministério público, prisão e reforma da justiça."
    ),
    "mar e pescas": (
        "Notícias sobre o setor marítimo e das pescas em Portugal, "
        "incluindo pesca artesanal e industrial, pescadores, aquacultura, "
        "gestão dos oceanos, zonas de pesca, política marítima e economia azul."
    ),
    "proteção social": (
        "Notícias sobre segurança social e proteção de cidadãos vulneráveis em Portugal, "
        "incluindo pensões, reformas, subsídios de desemprego, rendimento mínimo, "
        "prestações sociais, pobreza, exclusão social e apoio a famílias carenciadas."
    ),
    "saúde": (
        "Notícias sobre saúde e o Serviço Nacional de Saúde em Portugal, "
        "incluindo hospitais, médicos, doenças, tratamentos, medicamentos, "
        "saúde mental, listas de espera, urgências e políticas de saúde pública."
    ),
    "segurança": (
        "Notícias sobre segurança pública e criminalidade em Portugal, "
        "incluindo operações policiais, GNR, PSP, crime organizado, narcotráfico, "
        "terrorismo, proteção civil, bombeiros e acidentes."
    ),
    "trabalho": (
        "Notícias sobre emprego e mercado de trabalho em Portugal, "
        "incluindo salários, salário mínimo, sindicatos, greves, contratos de trabalho, "
        "teletrabalho, despedimentos e condições laborais."
    ),
    "transportes": (
        "Notícias sobre transportes e mobilidade em Portugal, "
        "incluindo comboios, metro, aviação, aeroportos, autoestradas, "
        "transporte público, veículos elétricos, logística e infraestruturas rodoviárias."
    ),
    "demografia": (
        "Notícias sobre evolução da população e tendências demográficas em Portugal, "
        "incluindo natalidade, envelhecimento da população, emigração, imigração, "
        "esperança de vida, censos e distribuição populacional pelo território."
    ),
    "desigualdades": (
        "Notícias sobre desigualdade social e económica em Portugal, "
        "incluindo pobreza, discriminação, distribuição de rendimentos, "
        "exclusão social, igualdade de género e políticas de inclusão."
    ),
    "infraestruturas": (
        "Notícias sobre construção e reabilitação de infraestruturas públicas em Portugal, "
        "incluindo estradas, pontes, saneamento, redes elétricas, fibra ótica, "
        "investimento público, PRR e modernização de aeroportos e portos."
    ),
    "relações internacionais": (
        "Notícias sobre política externa e diplomacia de Portugal no mundo, "
        "incluindo União Europeia, NATO, ONU, acordos bilaterais, cimeiras internacionais, "
        "lusofonia, CPLP, geopolítica e ajuda humanitária."
    ),
    "território": (
        "Notícias sobre desenvolvimento regional e coesão territorial em Portugal, "
        "incluindo assimetrias entre interior e litoral, ordenamento do território, "
        "fundos europeus, autarquias, ilhas dos Açores e Madeira e política regional."
    ),
}

CATEGORIAS = list(DESCRICOES.keys())

# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

def classificar_e5(df_gold: pd.DataFrame) -> dict:
    print(f"A carregar modelo: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Dimensões: {model.get_sentence_embedding_dimension()}")
    print(f"Max seq length: {model.max_seq_length}")

    # e5 requer prefixo "passage: " para documentos e "query: " para queries
    descricoes_lista = ["passage: " + DESCRICOES[c] for c in CATEGORIAS]
    noticias_lista   = ["query: " + t for t in df_gold["noticia_norm"].astype(str).fillna("").tolist()]

    print("A calcular embeddings das descrições...")
    emb_descricoes = model.encode(descricoes_lista, show_progress_bar=True,
                                  normalize_embeddings=True, batch_size=32)

    print("A calcular embeddings das notícias...")
    emb_noticias = model.encode(noticias_lista, show_progress_bar=True,
                                normalize_embeddings=True, batch_size=32)

    # Similaridade cosseno (240 x 24)
    sims = cosine_similarity(emb_noticias, emb_descricoes)

    # Normalizar para pseudo-probabilidades
    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs_matrix = sims / row_sums

    ids = df_gold["id"].astype(str).tolist()
    resultados = {}

    for i, nid in enumerate(ids):
        probs_row = probs_matrix[i]
        best_idx  = int(np.argmax(probs_row))
        cat_pred  = CATEGORIAS[best_idx]
        probs_dict = {
            cat: round(float(probs_row[j]), 6)
            for j, cat in enumerate(CATEGORIAS)
        }
        resultados[nid] = {
            "categoria": cat_pred,
            "probs":     probs_dict,
        }

    return resultados


def main():
    print(f"A carregar golden set: {GOLD_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Golden set não encontrado em {GOLD_PATH}")

    df_gold = pd.read_parquet(GOLD_PATH)
    df_gold["id"] = df_gold["id"].astype(str)
    print(f"Golden set: {df_gold.shape[0]} notícias, {df_gold['Categoria'].nunique()} categorias")

    resultados = classificar_e5(df_gold)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados em: {OUT_FILE}")

    # Preview + comparação rápida com modelo anterior
    from sklearn.metrics import accuracy_score, f1_score
    y_true = df_gold["Categoria"].tolist()
    y_pred = [resultados[nid]["categoria"] for nid in df_gold["id"].tolist()]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== multilingual-e5-large ===")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  F1-Macro:  {f1*100:.1f}%")
    print(f"\n  (modelo anterior: Accuracy 90.8%, F1-Macro 90.8%)")

    print("\n--- Preview (5 notícias) ---")
    for nid, res in list(resultados.items())[:5]:
        row = df_gold[df_gold["id"] == nid].iloc[0]
        top3 = sorted(res["probs"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nID {nid}: {row['titulo'][:60]}...")
        print(f"  Humano:  {row['Categoria']}")
        print(f"  Predito: {res['categoria']}")
        print(f"  Top-3:   {top3}")


if __name__ == "__main__":
    main()
