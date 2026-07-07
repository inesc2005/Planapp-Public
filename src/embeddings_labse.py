"""
Embeddings com LaBSE (Language-agnostic BERT Sentence Embeddings)
==================================================================
Classifica as 240 notícias do golden set usando o modelo
sentence-transformers/LaBSE — treinado em 109 línguas, muito robusto
para português, 768 dims.

Output:
  data/evaluation/preds_embeddings_labse.json
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
OUT_FILE     = OUT_DIR / "preds_embeddings_labse.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/LaBSE"

# =============================================================================
# DESCRIÇÕES (palavras-chave — LaBSE é robusto a este formato)
# =============================================================================
DESCRICOES = {
    "agricultura e floresta": (
        "agricultura floresta incêndios florestais sapadores silvicultura "
        "cultivo colheita regadio agricultura familiar vinha olivicultura "
        "gestão florestal reflorestação sobreiro eucalipto pastagem pecuária "
        "agricultores cooperativas rurais solo fertilizantes"
    ),
    "administração publica": (
        "administração pública função pública serviços públicos estado governo "
        "ministério secretaria organismos públicos burocracia concursos públicos "
        "funcionários públicos reforma administrativa descentralização autarquias "
        "municípios câmara municipal junta de freguesia serviço público"
    ),
    "ambiente e clima": (
        "ambiente clima alterações climáticas emissões carbono sustentabilidade "
        "energia renovável poluição resíduos reciclagem biodiversidade ecologia "
        "aquecimento global seca cheias inundações qualidade ar água proteção ambiental "
        "acordo de paris metas climáticas transição energética"
    ),
    "cultura": (
        "cultura arte música cinema teatro literatura museus exposições festivais "
        "património cultural artistas escritores realizadores monumentos arqueologia "
        "dança escultura pintura fotografia arquitetura criatividade espetáculos "
        "livros publicações prémios literários cultura portuguesa"
    ),
    "defesa": (
        "defesa forças armadas exército marinha força aérea NATO militares "
        "segurança nacional missões internacionais armamento cooperação militar "
        "defesa nacional ministério da defesa tropas operações militares "
        "orçamento defesa aliança atlântica paz segurança europeia"
    ),
    "desporto": (
        "desporto futebol basquetebol atletismo natação ciclismo ténis voleibol "
        "clubes desportivos federações campeonato liga primeira divisão seleção nacional "
        "jogadores treinadores árbitros estádios competições olímpicos paralímpicos "
        "transferências resultados classificações recordes"
    ),
    "economia": (
        "economia crescimento PIB recessão inflação mercado bolsa investimento "
        "exportações importações comércio empresa empresas multinacionais startups "
        "banco crédito juro taxa câmbio euro dívida pública orçamento finanças "
        "produtividade competitividade emprego desemprego mercado trabalho"
    ),
    "educação e formação": (
        "educação escolas ensino alunos professores universidade formação "
        "licenciatura mestrado doutoramento currículo exames avaliação "
        "ministério educação ensino superior básico secundário profissional "
        "literacia abandono escolar inclusão escolar tecnologia educação"
    ),
    "energia": (
        "energia solar eólica hídrica renovável eletricidade gás natural petróleo "
        "combustível tarifas energéticas produção energética redes elétricas "
        "transição energética descarbonização centrais elétricas barragens "
        "eficiência energética consumo energético EDP REN operadores"
    ),
    "habitação": (
        "habitação arrendamento imobiliário casa alojamento renda preço "
        "mercado imobiliário construção obras habitação pública apoios habitação "
        "crédito habitação banco hipoteca imóvel apartamento reabilitação urbana "
        "sem abrigo despejos alojamento local turistificação"
    ),
    "impostos": (
        "impostos IRS IRC IVA taxa fiscal fisco autoridade tributária receita fiscal "
        "contribuintes declaração impostos benefícios fiscais isenção deduções "
        "fraude fiscal evasão fiscal orçamento receitas tributárias regime fiscal "
        "imposto verde imposto automóvel IMI IMT reforma fiscal"
    ),
    "i&d e inovação": (
        "inovação investigação desenvolvimento tecnologia startup ciência "
        "inteligência artificial digitalização transformação digital "
        "investigadores laboratórios centros de investigação patentes "
        "fundos europeus FCT bolsas investigação publicações científicas "
        "robótica biotecnologia nanotecnologia spin-off transferência tecnologia"
    ),
    "justiça": (
        "justiça tribunal juiz magistrado procurador advogado crime lei "
        "sentença condenação absolvição recurso julgamento processo judicial "
        "ministério público policia judiciária corrupção fraude prisão "
        "código penal sistema judicial reforma judiciária acesso justiça"
    ),
    "mar e pescas": (
        "mar pesca pescadores pescas oceano litoral costa marinha recursos marinhos "
        "aquacultura porto pesqueiro barcos artes de pesca quotas pesca "
        "sustentabilidade oceanos biodiversidade marinha contaminação mar "
        "zona económica exclusiva política marítima economia azul"
    ),
    "proteção social": (
        "proteção social segurança social pensão reforma aposentação subsídio "
        "desemprego rendimento mínimo prestações sociais beneficiários vulneráveis "
        "pobreza exclusão social apoios sociais segurança social contribuições "
        "pensionistas idosos crianças famílias carenciadas assistência social"
    ),
    "saúde": (
        "saúde hospital médico enfermeiro SNS serviço nacional saúde doença "
        "tratamento medicamento vacina urgência consulta cirurgia doente "
        "saúde mental cancro diabetes cardiovascular pandemia covid epidemia "
        "cuidados primários centro saúde lista espera saúde pública"
    ),
    "segurança": (
        "segurança polícia GNR PSP crime criminalidade violência operação policial "
        "detenção arrestado investigação criminal droga narcotráfico terrorismo "
        "proteção civil bombeiros emergência acidente catástrofe segurança rodoviária "
        "ministério interior ordem pública segurança interna"
    ),
    "trabalho": (
        "trabalho emprego desemprego trabalhadores salário salário mínimo "
        "sindicatos greve negociação coletiva contrato trabalho despedimento "
        "teletrabalho condições trabalho mercado laboral recursos humanos "
        "ministério trabalho segurança social carreiras profissionais"
    ),
    "transportes": (
        "transportes mobilidade comboio metro autocarro avião aeroporto "
        "infraestrutura rodoviária autoestrada ferrovia tráfego CP TAP "
        "transporte público carros elétricos bicicleta partilhada "
        "logística portos mobilidade urbana planeamento transportes"
    ),
    "demografia": (
        "demografia população natalidade mortalidade envelhecimento emigração "
        "imigração taxa fertilidade esperança vida censos "
        "crescimento populacional distribuição população território "
        "jovens idosos família estrutura etária migrações fluxos migratórios"
    ),
    "desigualdades": (
        "desigualdades pobreza exclusão social desigualdade rendimentos "
        "coeficiente gini classes sociais discriminação género raça etnia "
        "acessibilidade oportunidades igualdade equidade distribuição riqueza "
        "vulnerabilidade social mobilidade social inclusão combate pobreza"
    ),
    "infraestruturas": (
        "infraestruturas obras públicas construção estradas pontes viadutos "
        "saneamento água abastecimento rede elétrica telecomunicações fibra ótica "
        "investimento público PRR plano recuperação resiliência projetos "
        "reabilitação infraestrutura aeroporto porto ferroviário modernização"
    ),
    "relações internacionais": (
        "relações internacionais diplomacia política externa embaixada consulado "
        "União Europeia NATO ONU cooperação internacional acordos bilaterais "
        "cimeiras presidência europeia lusofonia CPLP relações bilaterais "
        "geopolítica conflito internacional ajuda humanitária política global"
    ),
    "território": (
        "território regional desenvolvimento regional coesão territorial "
        "interior litoral assimetrias regionais fundos europeus CCDR "
        "ordenamento território planeamento urbano área metropolitana "
        "municípios autarquias descentralização política regional ilhas açores madeira"
    ),
}

CATEGORIAS = list(DESCRICOES.keys())

# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

def classificar_labse(df_gold: pd.DataFrame) -> dict:
    print(f"A carregar modelo: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Dimensões: {model.get_sentence_embedding_dimension()}")

    descricoes_lista = [DESCRICOES[c] for c in CATEGORIAS]
    noticias_lista   = df_gold["noticia_norm"].astype(str).fillna("").tolist()

    print("A calcular embeddings das descrições...")
    emb_descricoes = model.encode(descricoes_lista, show_progress_bar=True,
                                  normalize_embeddings=True, batch_size=32)

    print("A calcular embeddings das notícias...")
    emb_noticias = model.encode(noticias_lista, show_progress_bar=True,
                                normalize_embeddings=True, batch_size=32)

    sims = cosine_similarity(emb_noticias, emb_descricoes)

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

    resultados = classificar_labse(df_gold)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados em: {OUT_FILE}")

    from sklearn.metrics import accuracy_score, f1_score
    y_true = df_gold["Categoria"].tolist()
    y_pred = [resultados[nid]["categoria"] for nid in df_gold["id"].tolist()]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== LaBSE ===")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  F1-Macro:  {f1*100:.1f}%")
    print(f"\n  (paraphrase-mpnet: 90.8% | bge-m3: 85.1% | e5-large: 81.2%)")

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
