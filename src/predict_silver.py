"""
Predição em Amostra do Silver (dados novos, fora do golden set)
===============================================================
Amostra 50 artigos do silver que não estão no golden set,
corre os 5 modelos e guarda resultados para visualização na app.

Output:
  data/evaluation/preds_silver.json
  data/evaluation/silver_sample.parquet
"""

import re
import json
import time
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SILVER_PATH  = PROJECT_ROOT / "data" / "silver" / "cision_news_20251110.parquet"
GOLD_PATH    = PROJECT_ROOT / "data" / "gold" / "golden_set_10_por_categoria.parquet"
OUT_DIR      = PROJECT_ROOT / "data" / "evaluation"
OUT_PREDS    = OUT_DIR / "preds_silver.json"
OUT_SAMPLE   = OUT_DIR / "silver_sample.parquet"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
N_SAMPLE        = 50
RANDOM_SEED     = 42
MAX_BODY_CHARS  = 1500
API_DELAY       = 1.0

LLM_MODELS = {
    "LLM 1B": "llama3.2:1b",
    "LLM 3B": "llama3.2:3b",
    "LLM 8B": "llama3.1:8b",
}

# =============================================================================
# TAXONOMIA (igual aos outros scripts)
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
# HELPERS LLM
# =============================================================================
def sem_acentos(texto: str) -> str:
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("ascii")

CATEGORIAS_NORM = {sem_acentos(c): c for c in CATEGORIAS}

def mapear_categoria(texto: str) -> str:
    texto = texto.lower().strip()
    if texto in CATEGORIAS:
        return texto
    texto_norm = sem_acentos(texto)
    if texto_norm in CATEGORIAS_NORM:
        return CATEGORIAS_NORM[texto_norm]
    for norm, oficial in CATEGORIAS_NORM.items():
        if norm in texto_norm or texto_norm in norm:
            return oficial
    return "desconhecido"

def parsear_resposta(resposta_raw: str) -> dict | None:
    match = re.search(r'\{.*\}', resposta_raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

def normalizar_scores(scores_raw: dict) -> dict:
    total = sum(scores_raw.values())
    if total == 0:
        n = len(scores_raw)
        return {k: round(1/n, 6) for k in scores_raw}
    return {k: round(v / total, 6) for k, v in scores_raw.items()}

def construir_prompt(titulo: str, corpo: str) -> str:
    corpo_truncado = corpo[:MAX_BODY_CHARS]
    cats_json = "\n".join([f'    "{c}": <inteiro 0-100>,' for c in CATEGORIAS])
    return f"""És um sistema de classificação de notícias portuguesas.

Analisa a notícia abaixo e responde EXCLUSIVAMENTE em JSON válido com este formato:
{{
  "categoria": "<nome exato da categoria>",
  "confianca": <inteiro 0-100>,
  "scores": {{
{cats_json}
  }}
}}

Regras:
- "categoria" deve ser EXATAMENTE um dos nomes da lista acima
- "confianca" é o score da categoria escolhida (0-100)
- "scores" é a relevância de CADA categoria para esta notícia (0-100)
- Não incluas mais nada além do JSON

TÍTULO: {titulo}

CORPO: {corpo_truncado}"""

# =============================================================================
# TF-IDF
# =============================================================================
def correr_tfidf(df: pd.DataFrame) -> dict:
    print("TF-IDF: a classificar...")
    descricoes_lista = [DESCRICOES[c] for c in CATEGORIAS]
    noticias_lista   = df["noticia_norm"].astype(str).fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    todos = descricoes_lista + noticias_lista
    vectorizer.fit(todos)

    tfidf_desc     = vectorizer.transform(descricoes_lista)
    tfidf_noticias = vectorizer.transform(noticias_lista)
    sims           = cosine_similarity(tfidf_noticias, tfidf_desc)

    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs_matrix = sims / row_sums

    resultados = {}
    for i, nid in enumerate(df["id"].astype(str)):
        probs_row = probs_matrix[i]
        best_idx  = int(np.argmax(probs_row))
        resultados[nid] = {
            "categoria": CATEGORIAS[best_idx],
            "probs": {c: round(float(probs_row[j]), 6) for j, c in enumerate(CATEGORIAS)},
        }
    print(f"  TF-IDF: {len(resultados)} artigos classificados")
    return resultados

# =============================================================================
# EMBEDDINGS
# =============================================================================
def correr_embeddings(df: pd.DataFrame) -> dict:
    print("Embeddings: a carregar modelo...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    descricoes_lista = [DESCRICOES[c] for c in CATEGORIAS]
    noticias_lista   = df["noticia_norm"].astype(str).fillna("").tolist()

    emb_desc     = model.encode(descricoes_lista, show_progress_bar=False, normalize_embeddings=True)
    print("  Embeddings: a codificar notícias...")
    emb_noticias = model.encode(noticias_lista, show_progress_bar=True, normalize_embeddings=True)

    sims = cosine_similarity(emb_noticias, emb_desc)
    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs_matrix = sims / row_sums

    resultados = {}
    for i, nid in enumerate(df["id"].astype(str)):
        probs_row = probs_matrix[i]
        best_idx  = int(np.argmax(probs_row))
        resultados[nid] = {
            "categoria": CATEGORIAS[best_idx],
            "probs": {c: round(float(probs_row[j]), 6) for j, c in enumerate(CATEGORIAS)},
        }
    print(f"  Embeddings: {len(resultados)} artigos classificados")
    return resultados

# =============================================================================
# EMBEDDINGS E5-LARGE
# =============================================================================
def correr_embeddings_e5(df: pd.DataFrame) -> dict:
    print("Embeddings E5-Large: a carregar modelo...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    descricoes_lista = ["passage: " + DESCRICOES[c] for c in CATEGORIAS]
    noticias_lista   = ["query: " + t for t in df["noticia_norm"].astype(str).fillna("").tolist()]

    emb_desc     = model.encode(descricoes_lista, show_progress_bar=False, normalize_embeddings=True, batch_size=32)
    print("  E5: a codificar notícias...")
    emb_noticias = model.encode(noticias_lista, show_progress_bar=True, normalize_embeddings=True, batch_size=32)

    sims = cosine_similarity(emb_noticias, emb_desc)
    row_sums = sims.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs_matrix = sims / row_sums

    resultados = {}
    for i, nid in enumerate(df["id"].astype(str)):
        probs_row = probs_matrix[i]
        best_idx  = int(np.argmax(probs_row))
        resultados[nid] = {
            "categoria": CATEGORIAS[best_idx],
            "probs": {c: round(float(probs_row[j]), 6) for j, c in enumerate(CATEGORIAS)},
        }
    print(f"  E5: {len(resultados)} artigos classificados")
    return resultados

# =============================================================================
# LLM
# =============================================================================
def correr_llm(df: pd.DataFrame, model_label: str, model_id: str, client: OpenAI) -> dict:
    print(f"{model_label}: a classificar {len(df)} artigos com '{model_id}'...")
    resultados = {}

    for i, row in enumerate(df.itertuples()):
        nid    = str(row.id)
        titulo = str(row.titulo)
        corpo  = str(row.noticia_norm)
        prompt = construir_prompt(titulo, corpo)

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            resposta_raw = response.choices[0].message.content.strip()
            parsed = parsear_resposta(resposta_raw)

            if parsed and "categoria" in parsed and "scores" in parsed:
                categoria   = mapear_categoria(parsed["categoria"])
                scores_raw  = {cat: int(parsed["scores"].get(cat, 0)) for cat in CATEGORIAS}
                probs       = normalizar_scores(scores_raw)
                confianca   = float(parsed.get("confianca", 0))
                resultados[nid] = {"categoria": categoria, "confianca": confianca, "probs": probs}
            else:
                resultados[nid] = {
                    "categoria": "desconhecido", "confianca": 0.0,
                    "probs": {cat: round(1/len(CATEGORIAS), 6) for cat in CATEGORIAS},
                }
        except Exception as e:
            print(f"\n  Erro em {nid}: {e}")
            resultados[nid] = {
                "categoria": "desconhecido", "confianca": 0.0,
                "probs": {cat: round(1/len(CATEGORIAS), 6) for cat in CATEGORIAS},
            }

        print(f"\r  {i+1}/{len(df)}", end="", flush=True)
        time.sleep(API_DELAY)

    print(f"\n  {model_label}: concluído")
    return resultados

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("A carregar silver e golden set...")
    silver = pd.read_parquet(SILVER_PATH)
    gold   = pd.read_parquet(GOLD_PATH)

    gold_ids   = set(gold["id"].astype(str))
    silver["id_str"] = silver["id"].astype(str)
    fora_gold  = silver[~silver["id_str"].isin(gold_ids)].copy()

    # Filtra artigos com texto válido
    fora_gold = fora_gold[fora_gold["noticia_norm"].astype(str).str.len() > 100]

    amostra = fora_gold.sample(n=N_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
    amostra.to_parquet(OUT_SAMPLE)
    print(f"Amostra de {len(amostra)} artigos guardada em {OUT_SAMPLE}")

    resultados = {}

    # TF-IDF
    resultados["TF-IDF"] = correr_tfidf(amostra)

    # Embeddings
    resultados["Embeddings Paraphrase"] = correr_embeddings(amostra)

    # Embeddings E5-Large
    resultados["Embeddings E5-Large"] = correr_embeddings_e5(amostra)

    # LLMs
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    for label, model_id in LLM_MODELS.items():
        resultados[label] = correr_llm(amostra, label, model_id, client)

    with open(OUT_PREDS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados em: {OUT_PREDS}")

    # Resumo de concordância
    print("\n=== CONCORDÂNCIA ENTRE MODELOS ===")
    modelos = list(resultados.keys())
    concordam = 0
    for nid in amostra["id"].astype(str):
        cats = [resultados[m][nid]["categoria"] for m in modelos]
        if len(set(cats)) == 1:
            concordam += 1
    print(f"Todos concordam: {concordam}/{N_SAMPLE} ({concordam/N_SAMPLE*100:.0f}%)")


if __name__ == "__main__":
    main()
