"""
Classificação com LLM Base — mistral:7b-text (Zero-Shot)
=========================================================
Usa um modelo base (sem instruction tuning) com prompt de completamento.
Em vez de instruções, o modelo recebe contexto + taxonomia com descrições
e completa naturalmente com a categoria.

Diferença do instruct:
  Instruct: "Classifica esta notícia. Responde em JSON..."
  Base:     "Notícia: [...] | Categoria: [modelo completa]"

Output:
  data/evaluation/preds_llm_mistral_base.json
"""

import re
import json
import time
import unicodedata
import pandas as pd
from pathlib import Path
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID        = "mistral:7b-text"
MODEL_LABEL     = "mistral_base"
API_DELAY       = 0.5
MAX_BODY_CHARS  = 1500

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
OUT_FILE     = OUT_DIR / f"preds_llm_{MODEL_LABEL}.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TAXONOMIA COM DESCRIÇÕES
# =============================================================================
TAXONOMIA = {
    "agricultura e floresta":    "cultivo, silvicultura, incêndios florestais, pecuária, regadio, cooperativas agrícolas",
    "administração publica":     "ministérios, câmaras municipais, funcionários públicos, serviços do Estado, burocracia",
    "ambiente e clima":          "alterações climáticas, emissões CO2, poluição, reciclagem, biodiversidade, COP",
    "cultura":                   "arte, música, cinema, teatro, literatura, museus, festivais, património cultural",
    "defesa":                    "forças armadas, exército, marinha, NATO, missões militares, armamento",
    "desporto":                  "futebol, atletismo, natação, clubes, campeonatos, jogadores, seleção nacional",
    "economia":                  "PIB, inflação, mercados financeiros, empresas, investimento, exportações, dívida pública",
    "educação e formação":       "escolas, universidades, professores, alunos, exames, ensino superior, literacia",
    "energia":                   "solar, eólica, eletricidade, gás natural, petróleo, tarifas, descarbonização, EDP",
    "habitação":                 "arrendamento, imobiliário, preços de casas, construção, crédito habitação, sem-abrigo",
    "impostos":                  "IRS, IRC, IVA, fisco, autoridade tributária, fraude fiscal, deduções",
    "i&d e inovação":            "investigação, startups, inteligência artificial, digitalização, patentes, FCT",
    "justiça":                   "tribunais, julgamentos, corrupção, magistrados, ministério público, prisão",
    "mar e pescas":              "pesca, pescadores, oceano, aquacultura, economia azul, zona marítima",
    "proteção social":           "pensões, reformas, subsídios, rendimento mínimo, pobreza, segurança social",
    "saúde":                     "SNS, hospitais, médicos, doenças, tratamentos, urgências, saúde mental",
    "segurança":                 "polícia, GNR, PSP, crime, narcotráfico, terrorismo, bombeiros, proteção civil",
    "trabalho":                  "emprego, salários, sindicatos, greves, contratos, teletrabalho, despedimentos",
    "transportes":               "comboios, metro, aviação, aeroportos, autoestradas, CP, TAP, mobilidade",
    "demografia":                "natalidade, envelhecimento, emigração, imigração, censos, esperança de vida",
    "desigualdades":             "pobreza, discriminação, distribuição de rendimentos, exclusão social, género",
    "infraestruturas":           "estradas, pontes, saneamento, fibra ótica, PRR, obras públicas, empreitadas",
    "relações internacionais":   "diplomacia, União Europeia, NATO, ONU, acordos bilaterais, lusofonia, CPLP",
    "território":                "desenvolvimento regional, interior, litoral, fundos europeus, autarquias, Açores, Madeira",
}

CATEGORIAS = list(TAXONOMIA.keys())
TAXONOMIA_STR = "\n".join([f"- {cat}: {desc}" for cat, desc in TAXONOMIA.items()])

# =============================================================================
# HELPERS
# =============================================================================

def sem_acentos(texto: str) -> str:
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("ascii")

CATEGORIAS_NORM = {sem_acentos(c): c for c in CATEGORIAS}

def mapear_categoria(texto: str) -> str:
    texto = texto.lower().strip()
    # remover pontuação e texto extra após a categoria
    texto = re.split(r'[\n\r\.\,\;\:\!\?]', texto)[0].strip()
    if texto in CATEGORIAS:
        return texto
    texto_norm = sem_acentos(texto)
    if texto_norm in CATEGORIAS_NORM:
        return CATEGORIAS_NORM[texto_norm]
    for norm, oficial in CATEGORIAS_NORM.items():
        if norm in texto_norm or texto_norm in norm:
            return oficial
    return "desconhecido"

# =============================================================================
# PROMPT — estilo completamento para modelo base
# =============================================================================

def construir_prompt(titulo: str, corpo: str) -> str:
    corpo_truncado = corpo[:MAX_BODY_CHARS]
    return f"""Taxonomia de categorias de notícias portuguesas:
{TAXONOMIA_STR}

---

Notícia 1:
Título: Governo aumenta verbas para o SNS em 2025
Corpo: O Ministério da Saúde anunciou hoje um reforço de 500 milhões de euros para o Serviço Nacional de Saúde. A medida visa reduzir as listas de espera e contratar mais médicos e enfermeiros para os hospitais públicos portugueses.
Categoria: saúde

---

Notícia 2:
Título: Seleção portuguesa vence França no Euro 2024
Corpo: A seleção nacional de futebol derrotou a França por 2-1 na fase de grupos do Campeonato Europeu. Cristiano Ronaldo marcou o segundo golo aos 78 minutos, garantindo os três pontos para Portugal.
Categoria: desporto

---

Notícia 3:
Título: {titulo}
Corpo: {corpo_truncado}
Categoria:"""

# =============================================================================
# CLASSIFICAÇÃO
# =============================================================================

def classificar(df_gold: pd.DataFrame, client: OpenAI) -> dict:
    checkpoint_path = OUT_FILE
    resultados = {}

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            resultados = json.load(f)
        print(f"Checkpoint: {len(resultados)} noticias ja classificadas")

    ids     = df_gold["id"].astype(str).tolist()
    titulos = df_gold["titulo"].astype(str).tolist()
    corpos  = df_gold["noticia_norm"].astype(str).tolist()
    total   = len(ids)

    print(f"A classificar {total} noticias com '{MODEL_ID}'...")

    for i, (nid, titulo, corpo) in enumerate(zip(ids, titulos, corpos)):
        if nid in resultados:
            continue

        prompt = construir_prompt(titulo, corpo)

        try:
            response = client.completions.create(
                model=MODEL_ID,
                prompt=prompt,
                max_tokens=20,
                temperature=0,
                stop=["\n", "---", "Notícia"],
            )
            resposta_raw = response.choices[0].text.strip()
            categoria = mapear_categoria(resposta_raw)

            # fallback: se nao reconheceu, tenta encontrar categoria no texto
            if categoria == "desconhecido":
                for cat in CATEGORIAS:
                    if sem_acentos(cat) in sem_acentos(resposta_raw.lower()):
                        categoria = cat
                        break

            resultados[nid] = {
                "categoria": categoria,
                "confianca": 100.0 if categoria != "desconhecido" else 0.0,
                "probs":     {cat: (1.0 if cat == categoria else 0.0) for cat in CATEGORIAS},
                "raw":       resposta_raw,
            }

        except Exception as e:
            print(f"\nErro noticia {nid}: {e}")
            resultados[nid] = {
                "categoria": "desconhecido",
                "confianca": 0.0,
                "probs":     {cat: round(1/len(CATEGORIAS), 6) for cat in CATEGORIAS},
                "raw":       "ERRO",
            }

        done = i + 1
        ok   = sum(1 for v in resultados.values() if v["categoria"] != "desconhecido")
        print(f"\r  {done}/{total} ({done/total*100:.0f}%)  OK: {ok}", end="", flush=True)

        if done % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
            print(f"\n  Checkpoint guardado ({done}/{total})")

        time.sleep(API_DELAY)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    ok = sum(1 for v in resultados.values() if v["categoria"] != "desconhecido")
    print(f"\n\nConcluido -> {checkpoint_path}  ({ok}/{total} OK)")
    return resultados


def main():
    print(f"A carregar golden set: {GOLD_PATH}")
    df_gold = pd.read_parquet(GOLD_PATH)
    df_gold["id"] = df_gold["id"].astype(str)
    print(f"Golden set: {df_gold.shape[0]} noticias, {df_gold['Categoria'].nunique()} categorias")

    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)

    resultados = classificar(df_gold, client)

    from sklearn.metrics import accuracy_score, f1_score
    y_true   = df_gold["Categoria"].tolist()
    y_pred   = [resultados[nid]["categoria"] for nid in df_gold["id"].tolist()]
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acertos  = sum(p == t for p, t in zip(y_pred, y_true))

    print(f"\n=== mistral:7b-text (base) ===")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  F1-Macro:  {f1*100:.1f}%")
    print(f"  Acertos:   {acertos}/240")
    print(f"\n  (llama3.1:8b instruct: 62.5% | llama3.2:3b: 54.2% | llama3.2:1b: 33.2%)")

    # Preview
    print("\n--- Preview (5 noticias) ---")
    for nid, res in list(resultados.items())[:5]:
        row = df_gold[df_gold["id"] == nid].iloc[0]
        print(f"\nID {nid}: {row['titulo'][:60]}...")
        print(f"  Humano:  {row['Categoria']}")
        print(f"  Predito: {res['categoria']}  raw: '{res['raw']}'")


if __name__ == "__main__":
    main()
