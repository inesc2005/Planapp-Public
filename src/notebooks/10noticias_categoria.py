import pandas as pd
from pathlib import Path

print("A carregar dataset silver...")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def find_project_root(start: Path, marker: str = "data") -> Path:
    current = start
    for _ in range(6):
        if (current / marker).exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        f"Não consegui encontrar a pasta '{marker}' subindo a partir de {start}."
    )

PROJECT_ROOT = find_project_root(BASE_DIR)

input_path = PROJECT_ROOT / "data" / "silver" / "cision_news_20251110.parquet"
output_dir = PROJECT_ROOT / "data" / "gold"
output_dir.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("Input path:  ", input_path)
print("Existe ficheiro?", input_path.exists())

if not input_path.exists():
    raise FileNotFoundError(f"\n❌ Ficheiro silver não encontrado em:\n  {input_path}")

df = pd.read_parquet(input_path)
print("\nDataset carregado com sucesso.")
print("Dimensão original:", df.shape)

# ─────────────────────────────────────────────
# GOLDEN SET — ordem manual das categorias e notícias
# ─────────────────────────────────────────────
categorias = [
    ("agricultura e floresta", [
        117473737, 116743046, 117729727, 116460264, 118761156,
        117468688, 118781158, 119784497, 119114345, 119415791,
    ]),
    ("administração publica", [
        119848198, 119806417, 119856314, 116778772, 117693697,
        119008632, 119327429, 117538813, 117277538, 119092724,
    ]),
    ("ambiente e clima", [
        119563198, 117592370, 117511694, 118161322, 118818328,
        117292440, 119278718, 117524880, 118527554, 119905711,
    ]),
    ("cultura", [
        119172159, 119905597, 119924198, 116990217, 116589381,
        116818492, 117596591, 117040794, 118593932, 118361891,
    ]),
    ("defesa", [
        119328435, 119338247, 118781404, 118128738, 117695551,
        119803170, 118050323, 117835685, 117601556, 117805416,
    ]),
    ("desporto", [
        119910091, 119732632, 118472505, 117427360, 117522542,
        118991530, 118261765, 117881122, 118186057, 119261741,
    ]),
    ("economia", [
        116960074, 119864892, 119827973, 119750985, 116964479,
        117765706, 118448744, 118899198, 116610824, 116610683,
    ]),
    ("educação e formação", [
        119221528, 117788218, 117434653, 118014820, 118014778,
        119501987, 119192344, 117789914, 118251951, 116741961,
    ]),
    ("energia", [
        118527607, 118573995, 119046203, 118168795, 118301900,
        119208215, 118283720, 119258086, 116892174, 116930101,
    ]),
    ("habitação", [
        119095915, 119390141, 117272281, 117789736, 117441285,
        117177763, 118362103, 119501765, 119217419, 118128845,
    ]),
    ("impostos", [
        119914542, 119536445, 118002166, 119536229, 119535030,
        119951234, 119290681, 119463480, 117771465, 119608747,
    ]),
    ("i&d e inovação", [
        119520225, 116674759, 118207874, 117617992, 117448283,
        118975822, 117862804, 119080737, 119186072, 119950007,
    ]),
    ("justiça", [
        119168648, 118901517, 119846767, 119851178, 118148202,
        117706894, 119677366, 118104562, 118976449, 119955773,
    ]),
    ("mar e pescas", [
        118992163, 117975334, 119558519, 118274301, 116930532,
        117418263, 118793129, 117507037, 117688003, 118093100,
    ]),
    ("proteção social", [
        119017291, 119857754, 119061709, 117693075, 119559206,
        119569896, 118268624, 118001138, 119614679, 117766410,
    ]),
    ("saúde", [
        119258353, 119410587, 119501840, 119823456, 119785362,
        119814395, 119114046, 119409021, 119628903, 116435101,
    ]),
    ("segurança", [
        119229913, 116729348, 117992954, 119258262, 118186517,
        118151421, 116754318, 116932484, 116727093, 117798899,
    ]),
    ("trabalho", [
        119076493, 118499757, 119040242, 119235404, 116682400,
        119305483, 119786380, 118978405, 119566942, 119841639,
    ]),
    ("transportes", [
        119906018, 118460082, 116615102, 119849410, 116836150,
        118457817, 119076635, 118899366, 119298943, 117049211,
    ]),
    ("demografia", [
        116496305, 118449249, 117926260, 119572214, 119677663,
        117463506, 119172114, 119294545, 118795192, 118434152,
    ]),
    ("desigualdades", [
        117435723, 117432863, 117841479, 119731119, 119669362,
        119671216, 118870377, 119886682, 118615387, 117821126,
    ]),
    ("infraestruturas", [
        117747228, 119944534, 117349703, 119063347, 117445592,
        117899268, 118269155, 119457447, 117903353, 118208190,
    ]),
    ("relações internacionais", [
        117505199, 117693043, 119485073, 119795004, 117676741,
        116699509, 118879956, 118610285, 117814206, 118281556,
    ]),
    ("território", [
        117260490, 117260345, 117260492, 118115025, 119236090,
        116965361, 116714927, 118686096, 119841421, 117507259,
    ]),
]

# ─────────────────────────────────────────────
# MAPA ID → categoria e posição de ordenação
# ─────────────────────────────────────────────
id_to_cat = {}
order_map = {}
pos = 0
for cat_name, ids in categorias:
    for news_id in ids:
        id_to_cat[news_id] = cat_name
        order_map[news_id] = pos
        pos += 1

all_ids = list(id_to_cat.keys())
print(f"\nTotal de IDs esperados: {len(all_ids)} ({len(categorias)} categorias × 10)")

# ─────────────────────────────────────────────
# DETETAR COLUNA DO ID
# ─────────────────────────────────────────────
id_col = next((c for c in ["id", "ID", "news_id", "article_id"] if c in df.columns), None)
if id_col is None:
    raise ValueError(f"Coluna de ID não encontrada. Disponíveis: {list(df.columns)}")
print(f"Coluna de ID: '{id_col}'")

df[id_col] = pd.to_numeric(df[id_col], errors="coerce")

# ─────────────────────────────────────────────
# FILTRAR, ORDENAR PELA POSIÇÃO MANUAL
# ─────────────────────────────────────────────
gold_df = df[df[id_col].isin(all_ids)].copy()
gold_df["Categoria"] = gold_df[id_col].map(id_to_cat)
gold_df["_order"]    = gold_df[id_col].map(order_map)
gold_df = gold_df.sort_values("_order").reset_index(drop=True)
gold_df = gold_df.drop(columns=["_order"])

# ─────────────────────────────────────────────
# VERIFICAR IDs EM FALTA
# ─────────────────────────────────────────────
ids_encontrados = set(gold_df[id_col].tolist())
ids_em_falta = sorted(set(all_ids) - ids_encontrados)

if ids_em_falta:
    print(f"\n⚠️  {len(ids_em_falta)} IDs não encontrados:")
    for mid in ids_em_falta:
        print(f"   id={mid}  categoria={id_to_cat[mid]}")
else:
    print("\n✅ Todos os IDs foram encontrados.")

# ─────────────────────────────────────────────
# COLUNAS FINAIS
# ─────────────────────────────────────────────
title_col = next((c for c in ["titulo", "title", "headline"] if c in gold_df.columns), None)
body_col  = next((c for c in ["noticia_norm", "corpo", "body", "text", "noticia_raw", "noticia"] if c in gold_df.columns), None)

if not title_col:
    raise ValueError(f"Coluna de título não encontrada. Disponíveis: {list(gold_df.columns)}")
if not body_col:
    raise ValueError(f"Coluna de corpo não encontrada. Disponíveis: {list(gold_df.columns)}")

gold_df = gold_df[[id_col, title_col, body_col, "Categoria"]].copy()
gold_df.columns = ["id", "titulo", "noticia_norm", "Categoria"]

# ─────────────────────────────────────────────
# GUARDAR
# ─────────────────────────────────────────────
csv_path     = output_dir / "golden_set_10_por_categoria.csv"
parquet_path = output_dir / "golden_set_10_por_categoria.parquet"

gold_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
gold_df.to_parquet(parquet_path, index=False)

print("\n" + "=" * 60)
print("✅ Dataset gold criado com sucesso!")
print(f"   CSV:     {csv_path}")
print(f"   Parquet: {parquet_path}")
print(f"   Shape:   {gold_df.shape}")
print("=" * 60)

print("\nOrdem e contagem por categoria:")
for cat_name, ids in categorias:
    n = (gold_df["Categoria"] == cat_name).sum()
    status = "✅" if n == 10 else f"❌ ({n})"
    print(f"  {status}  {cat_name}")