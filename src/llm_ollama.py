"""
Classificação LLM via Ollama (Zero-Shot)
=========================================
Classifica as 240 notícias do golden set usando modelos Llama via Ollama.
Endpoint local compatível com OpenAI: http://localhost:11434

Modelos configurados:
  llm_1b  -> llama3.2:1b
  llm_2b  -> gemma2:2b
  llm_3b  -> llama3.2:3b
  llm_8b  -> llama3.1:8b

Uso:
  python src/llm_ollama.py
  python src/llm_ollama.py --modelo llm_8b --model-id llama3.1:8b

Output:
  data/evaluation/preds_llm_{model_label}.json
"""

import re
import json
import time
import argparse
import unicodedata
import pandas as pd
from pathlib import Path
from openai import OpenAI

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Modelos a correr — edita aqui para adicionar/remover
LLM_MODELS = {
    "llm_1b": "llama3.2:1b",
    "llm_3b": "llama3.2:3b",
    "llm_8b": "llama3.1:8b",
}

API_DELAY      = 1.0
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
GOLD_PATH    = PROJECT_ROOT / "data" / "gold" / "golden_set_10_por_categoria.parquet"
OUT_DIR      = PROJECT_ROOT / "data" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TAXONOMIA
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

# =============================================================================
# HELPERS
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

# =============================================================================
# PROMPT
# =============================================================================

def construir_prompt(titulo: str, corpo: str) -> str:
    corpo_truncado = corpo[:MAX_BODY_CHARS]
    return f"""És um sistema de classificação de notícias portuguesas.

Analisa a notícia abaixo e responde EXCLUSIVAMENTE em JSON válido com este formato:
{{
  "categoria": "<nome exato da categoria>",
  "confianca": <inteiro 0-100>,
  "scores": {{
    "agricultura e floresta": <inteiro 0-100>,
    "administração publica": <inteiro 0-100>,
    "ambiente e clima": <inteiro 0-100>,
    "cultura": <inteiro 0-100>,
    "defesa": <inteiro 0-100>,
    "desporto": <inteiro 0-100>,
    "economia": <inteiro 0-100>,
    "educação e formação": <inteiro 0-100>,
    "energia": <inteiro 0-100>,
    "habitação": <inteiro 0-100>,
    "impostos": <inteiro 0-100>,
    "i&d e inovação": <inteiro 0-100>,
    "justiça": <inteiro 0-100>,
    "mar e pescas": <inteiro 0-100>,
    "proteção social": <inteiro 0-100>,
    "saúde": <inteiro 0-100>,
    "segurança": <inteiro 0-100>,
    "trabalho": <inteiro 0-100>,
    "transportes": <inteiro 0-100>,
    "demografia": <inteiro 0-100>,
    "desigualdades": <inteiro 0-100>,
    "infraestruturas": <inteiro 0-100>,
    "relações internacionais": <inteiro 0-100>,
    "território": <inteiro 0-100>
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
# CLASSIFICAÇÃO
# =============================================================================

def classificar_llm(df_gold: pd.DataFrame, model_id: str, model_label: str, client: OpenAI) -> dict:
    checkpoint_path = OUT_DIR / f"preds_llm_{model_label}.json"

    resultados = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            resultados = json.load(f)
        print(f"[{model_label}] Checkpoint: {len(resultados)} notícias já classificadas")

    ids     = df_gold["id"].astype(str).tolist()
    titulos = df_gold["titulo"].astype(str).tolist()
    corpos  = df_gold["noticia_norm"].astype(str).tolist()

    total = len(ids)
    print(f"\n[{model_label}] A classificar {total} notícias com '{model_id}'...")

    for i, (nid, titulo, corpo) in enumerate(zip(ids, titulos, corpos)):
        if nid in resultados:
            continue

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
                categoria = mapear_categoria(parsed["categoria"])
                if categoria == "desconhecido":
                    scores = parsed.get("scores", {})
                    if scores:
                        best_raw = max(scores, key=scores.get)
                        categoria = mapear_categoria(best_raw)

                scores_raw = {cat: int(parsed["scores"].get(cat, 0)) for cat in CATEGORIAS}
                probs      = normalizar_scores(scores_raw)
                confianca  = float(parsed.get("confianca", 0))

                resultados[nid] = {
                    "categoria": categoria,
                    "confianca": confianca,
                    "probs":     probs,
                    "raw":       resposta_raw,
                }
            else:
                resultados[nid] = {
                    "categoria": "desconhecido",
                    "confianca": 0.0,
                    "probs":     {cat: round(1/len(CATEGORIAS), 6) for cat in CATEGORIAS},
                    "raw":       resposta_raw,
                }

        except Exception as e:
            print(f"\nErro na noticia {nid}: {e}")
            resultados[nid] = {
                "categoria": "desconhecido",
                "confianca": 0.0,
                "probs":     {cat: round(1/len(CATEGORIAS), 6) for cat in CATEGORIAS},
                "raw":       "ERRO",
            }

        done = i + 1
        pct  = done / total * 100
        ok   = sum(1 for v in resultados.values() if v["categoria"] != "desconhecido")
        print(f"\r  {done}/{total} ({pct:.0f}%)  OK: {ok}", end="", flush=True)

        if done % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
            print(f"\n  Checkpoint guardado ({done}/{total})")

        time.sleep(API_DELAY)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    ok = sum(1 for v in resultados.values() if v["categoria"] != "desconhecido")
    print(f"\n[{model_label}] Concluido -> {checkpoint_path}  ({ok}/{total} OK)")

    return resultados

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classificação via Ollama")
    parser.add_argument("--modelo",   default=None, help="Label do modelo (ex: llm_8b)")
    parser.add_argument("--model-id", default=None, help="ID Ollama (ex: llama3.1:8b)")
    args = parser.parse_args()

    # Permite sobrepor modelo via linha de comandos
    modelos = dict(LLM_MODELS)
    if args.modelo and args.model_id:
        modelos = {args.modelo: args.model_id}

    print(f"A carregar golden set: {GOLD_PATH}")
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Golden set não encontrado em {GOLD_PATH}")

    df_gold = pd.read_parquet(GOLD_PATH)
    df_gold["id"] = df_gold["id"].astype(str)
    print(f"Golden set: {df_gold.shape[0]} notícias")

    client = OpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
    )

    for model_label, model_id in modelos.items():
        resultados = classificar_llm(df_gold, model_id, model_label, client)

        print(f"\n--- Preview {model_label} (3 notícias) ---")
        for nid, res in list(resultados.items())[:3]:
            row = df_gold[df_gold["id"] == nid].iloc[0]
            top3 = sorted(res["probs"].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\nID {nid}: {row['titulo'][:60]}...")
            print(f"  Humano:      {row['Categoria']}")
            print(f"  Predito:     {res['categoria']} (confiança: {res['confianca']:.0f})")
            print(f"  Top-3 probs: {top3}")


if __name__ == "__main__":
    main()
