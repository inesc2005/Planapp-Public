from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support

st.set_page_config(
    page_title="Planapp — Avaliação de Modelos",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; background: #F5F7FF !important; }
.main .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1300px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1B2B6B !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: #C7D4FF !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem !important; font-weight: 500 !important; padding: 8px 10px !important; border-radius: 6px !important; }

/* Headings */
h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #111827 !important; margin-bottom: 2px !important; }
h2, h3 { font-size: 0.75rem !important; font-weight: 700 !important; color: #6B7280 !important; text-transform: uppercase !important; letter-spacing: 1px !important; }

/* KPI card */
.kpi { background:#fff; border-radius:12px; padding:18px 22px; border:1px solid #E5E7EB; box-shadow:0 1px 4px rgba(0,0,0,.05); text-align:center; }
.kpi-label { font-size:.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#9CA3AF; margin-bottom:6px; }
.kpi-value { font-size:2.1rem; font-weight:700; color:#1B2B6B; font-family:'IBM Plex Mono',monospace; line-height:1; }
.kpi-winner { font-size:.68rem; color:#6B7280; margin-top:5px; }
.kpi-best { border-top:3px solid #2563EB !important; }

/* Inline badges */
.b { display:inline-block; padding:2px 9px; border-radius:20px; font-size:.7rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.b-ok  { background:#D1FAE5; color:#065F46; }
.b-err { background:#FEE2E2; color:#991B1B; }
.b-cat { background:#EFF6FF; color:#1D4ED8; }
.b-alt { background:#F5F3FF; color:#5B21B6; }

/* Probability bar mini */
.prob-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:.72rem; font-family:'IBM Plex Mono',monospace; }
.prob-label { width:160px; color:#374151; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prob-bar-wrap { flex:1; background:#F3F4F6; border-radius:4px; height:8px; }
.prob-bar { height:8px; border-radius:4px; }
.prob-val { width:36px; text-align:right; color:#6B7280; }

/* Expander */
[data-testid="stExpander"] { background:#fff !important; border:1px solid #E5E7EB !important; border-radius:10px !important; margin-bottom:6px !important; box-shadow:0 1px 3px rgba(0,0,0,.04) !important; }
[data-testid="stExpander"]:hover { border-color:#93C5FD !important; }
[data-testid="stExpander"] summary { font-size:.84rem !important; font-weight:500 !important; color:#1F2937 !important; }

/* Section card */
.card { background:#fff; border-radius:14px; padding:20px 24px; border:1px solid #E5E7EB; box-shadow:0 1px 4px rgba(0,0,0,.05); margin-bottom:18px; }
.section-label { font-size:.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:#2563EB; border-bottom:2px solid #EFF6FF; padding-bottom:8px; margin-bottom:14px; }

/* Stats row */
.stats { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:10px 16px; font-size:.82rem; color:#1E40AF; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODELOS = {
    "TF-IDF":               "cat_tfidf",
    "Embeddings Paraphrase": "cat_embeddings",
    "Embeddings E5-Large":  "cat_embeddings_e5",
    "Embeddings BGE-M3":    "cat_embeddings_bge_m3",
    "LLM 1B":               "cat_llm_llm_1b",
    "LLM 3B":               "cat_llm_llm_3b",
    "LLM 8B":               "cat_llm_llm_8b",
    "LLM Mistral":     "cat_llm_mistral_base",
}
PREDS_JSON = {
    "TF-IDF":               "data/evaluation/preds_tfidf.json",
    "Embeddings Paraphrase": "data/evaluation/preds_embeddings.json",
    "Embeddings E5-Large":  "data/evaluation/preds_embeddings_e5.json",
    "Embeddings BGE-M3":    "data/evaluation/preds_embeddings_bge_m3.json",
    "LLM 1B":               "data/evaluation/preds_llm_llm_1b.json",
    "LLM 3B":               "data/evaluation/preds_llm_llm_3b.json",
    "LLM 8B":               "data/evaluation/preds_llm_llm_8b.json",
    "LLM Mistral":     "data/evaluation/preds_llm_mistral_base.json",
}
CORES = {
    "TF-IDF":               "#2563EB",
    "Embeddings Paraphrase": "#0891B2",
    "Embeddings E5-Large":  "#059669",
    "Embeddings BGE-M3":    "#7C2D12",
    "LLM 1B":               "#D97706",
    "LLM 3B":               "#DC2626",
    "LLM 8B":               "#7C3AED",
    "LLM Mistral":     "#9333EA",
}
PLOTLY = dict(
    font_family="Inter, sans-serif", font_color="#374151",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
)

# ── Data ───────────────────────────────────────────────────────────────────────
def _fix_str(s: str) -> str:
    """Fix titles that are UTF-8 bytes read as latin-1 (mixed-encoding CSV)."""
    if not isinstance(s, str) or "Ã" not in s:
        return s  # no Ã pattern → already correct latin-1
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


@st.cache_data
def load():
    av = pd.read_csv("data/evaluation/avaliacao_final.csv", encoding="latin-1")
    av.columns = [c.lstrip("ï»¿") for c in av.columns]
    av["titulo"] = av["titulo"].apply(_fix_str)
    av["Categoria_humana"] = av["Categoria_humana"].apply(_fix_str)
    for col in [c for c in av.columns if c.startswith("cat_")]:
        av[col] = av[col].apply(_fix_str)

    tab = pd.read_csv("data/evaluation/tabela_comparativa.csv", encoding="cp1252")
    tab.columns = [c.lstrip("ï»¿") for c in tab.columns]
    def norm(n):
        s = n.lower()
        if "e5" in s:      return "Embeddings E5-Large"
        if "bge" in s:     return "Embeddings BGE-M3"
        if "embed" in s:   return "Embeddings Paraphrase"
        if "idf"   in s:   return "TF-IDF"
        if "mistral" in s: return "LLM Mistral"
        if "1b"    in s:   return "LLM 1B"
        if "3b"    in s:   return "LLM 3B"
        if "8b"    in s:   return "LLM 8B"
        return n
    tab["Modelo"] = tab["Modelo"].map(norm)

    preds = {}
    for nome, path in PREDS_JSON.items():
        with open(path, encoding="latin-1") as f:
            raw = json.load(f)
        # fix encoding in all string fields inside each prediction
        fixed = {}
        for art_id, info in raw.items():
            entry = dict(info)
            if "categoria" in entry:
                entry["categoria"] = _fix_str(entry["categoria"])
            if "probs" in entry:
                entry["probs"] = {_fix_str(k): v for k, v in entry["probs"].items()}
            fixed[art_id] = entry
        preds[nome] = fixed
    return av, tab, preds

@st.cache_data
def metricas_por_categoria(av):
    out = {}
    y_true = av["Categoria_humana"]
    for nome, col in MODELOS.items():
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, av[col], average=None, labels=sorted(y_true.unique())
        )
        out[nome] = pd.DataFrame({
            "categoria": sorted(y_true.unique()),
            "precision": p, "recall": r, "f1": f1,
        })
    return out

@st.cache_data
def load_gold_text():
    df = pd.read_parquet("data/gold/golden_set_10_por_categoria.parquet")
    df["id"] = df["id"].astype(str)
    df["noticia_norm"] = df["noticia_norm"].apply(_fix_str)
    return dict(zip(df["id"], df["noticia_norm"]))

@st.cache_data
def load_silver():
    sample_path = "data/evaluation/silver_sample.parquet"
    preds_path  = "data/evaluation/preds_silver.json"
    try:
        df = pd.read_parquet(sample_path)
        df["titulo"]      = df["titulo"].apply(_fix_str)
        df["noticia_norm"] = df["noticia_norm"].apply(_fix_str)
        with open(preds_path, encoding="utf-8") as f:
            raw = json.load(f)
        fixed = {}
        for modelo, arts in raw.items():
            fixed[modelo] = {}
            for art_id, info in arts.items():
                entry = dict(info)
                entry["categoria"] = _fix_str(entry.get("categoria", ""))
                if "probs" in entry:
                    entry["probs"] = {_fix_str(k): v for k, v in entry["probs"].items()}
                fixed[modelo][art_id] = entry
        return df, fixed
    except FileNotFoundError:
        return None, None

av, tab, preds = load()
met_cat = metricas_por_categoria(av)
gold_text = load_gold_text()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 4px 8px'>
      <div style='font-size:1.25rem;font-weight:700;color:#fff;letter-spacing:-.3px'>Planapp</div>
      <div style='font-size:.7rem;color:#93A8D4;margin-top:3px'></div>
    </div>
    <hr style='border-color:rgba(255,255,255,.1);margin:10px 0'>
    """, unsafe_allow_html=True)
    pagina = st.radio("", ["Comparação de Modelos", "Explorador de Notícias", "Notícias Novas"],
                      label_visibility="collapsed")

# ── Helpers ────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", best=False):
    cls = "kpi kpi-best" if best else "kpi"
    st.markdown(f"""
    <div class='{cls}'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-winner'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def prob_bars(probs: dict, correct_cat: str, pred_cat: str = None, top=3):
    top_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top]
    mx = top_items[0][1] if top_items else 1
    rows = ""
    for cat, val in top_items:
        bar_pct = (val / mx * 100) if mx > 0 else 0
        display_pct = val * 100  # probs are 0-1 scale
        color = "#2563EB" if cat == (pred_cat if pred_cat else correct_cat) else "#D1D5DB"
        rows += f"""
        <div class='prob-row'>
          <div class='prob-label'>{cat}</div>
          <div class='prob-bar-wrap'><div class='prob-bar' style='width:{bar_pct:.0f}%;background:{color}'></div></div>
          <div class='prob-val'>{display_pct:.1f}%</div>
        </div>"""
    st.markdown(rows, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — COMPARAÇÃO DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Comparação de Modelos":
    st.title("Comparação de Modelos")
    st.caption("8 modelos avaliados em 240 artigos — 10 por categoria × 24 categorias")

    df = tab.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Macro"]:
        df[c] = df[c].str.replace("%", "").astype(float)
    df["Acertos_n"] = df["Acertos"].str.split("/").str[0].astype(int)
    melhor = df.loc[df["F1-Macro"].idxmax()]

    # ── Selector de modelo (tipo PowerBI) ──
    if "modelo_sel_comp" not in st.session_state:
        st.session_state["modelo_sel_comp"] = None

    st.markdown("**Seleciona um modelo para destacar nos gráficos**")

    # label geral e nome específico por baixo
    MODELO_INFO = {
        "Todos":                ("Todos",      ""),
        "TF-IDF":               ("TF-IDF",     "TF-IDF Descrições"),
        "Embeddings Paraphrase":("Embeddings", "paraphrase-mpnet-v2"),
        "Embeddings E5-Large":  ("Embeddings", "multilingual-e5-large"),
        "Embeddings BGE-M3":    ("Embeddings", "bge-m3"),
        "LLM 1B":               ("LLM",        "llama3.2:1b"),
        "LLM 3B":               ("LLM",        "llama3.2:3b"),
        "LLM 8B":               ("LLM",        "llama3.1:8b"),
        "LLM Mistral":     ("LLM",   "mistral:7b-text"),
    }

    todos_modelos = ["Todos"] + list(MODELOS.keys())
    btn_cols = st.columns(len(todos_modelos))

    for i, nome in enumerate(todos_modelos):
        label_geral, label_esp = MODELO_INFO[nome]
        is_sel = (st.session_state["modelo_sel_comp"] is None and nome == "Todos") or \
                 (st.session_state["modelo_sel_comp"] == nome)
        with btn_cols[i]:
            if st.button(label_geral, type="primary" if is_sel else "secondary",
                         use_container_width=True, key=f"btn_{nome}"):
                st.session_state["modelo_sel_comp"] = None if nome == "Todos" else nome
                st.rerun()
            if label_esp:
                st.markdown(
                    f"<div style='text-align:center;font-size:.8rem;color:#6B7280;"
                    f"margin-top:-8px;font-family:IBM Plex Mono,monospace'>{label_esp}</div>",
                    unsafe_allow_html=True,
                )

    sel = st.session_state["modelo_sel_comp"]  # None ou nome do modelo

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── KPI cards — mudam com o modelo selecionado ──
    if sel is None:
        best_f1 = df.loc[df["F1-Macro"].idxmax()]
        media_f1 = df["F1-Macro"].mean()
        kpi_f1_val  = f"{best_f1['F1-Macro']:.1f}%"
        kpi_f1_sub  = f"Melhor: {best_f1['Modelo']}"
        kpi_ac_val  = f"{media_f1:.1f}%"
        kpi_ac_sub  = "Média F1-Macro dos 7 modelos"
        kpi_best_f1 = True
    else:
        row_sel = df[df["Modelo"] == sel].iloc[0]
        kpi_f1_val  = f"{row_sel['F1-Macro']:.1f}%"
        kpi_f1_sub  = f"{sel}"
        kpi_ac_val  = row_sel["Acertos"]
        kpi_ac_sub  = f"Accuracy: {row_sel['Accuracy']:.1f}%"
        kpi_best_f1 = (sel == melhor["Modelo"])

    kpi_c2_label = "Acertos" if sel else "Média F1-Macro"
    c1, c2, c_space = st.columns([1, 1, 2])
    with c1:
        kpi_card("F1-Macro", kpi_f1_val, kpi_f1_sub, best=kpi_best_f1)
    with c2:
        kpi_card(kpi_c2_label, kpi_ac_val, kpi_ac_sub, best=False)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── helpers de cor: destaca modelo selecionado, esmaece os outros ──
    def cor_modelo(nome):
        if sel is None or nome == sel:
            return CORES.get(nome, "#94A3B8")
        return "#D1D5DB"

    def opacidade_modelo(nome):
        if sel is None or nome == sel:
            return 1.0
        return 0.25

    # ── Tabela + Precision vs Recall ──
    col_t, col_g = st.columns([1, 1.6], gap="large")

    with col_t:
        st.markdown("**Resultados Globais**")
        df_show = df[["Modelo", "Accuracy", "Precision", "Recall", "F1-Macro", "Acertos"]].copy()
        for c in ["Accuracy", "Precision", "Recall", "F1-Macro"]:
            df_show[c] = df_show[c].map("{:.1f}%".format)

        def hl(row):
            if sel is not None:
                if row["Modelo"] == sel:
                    return ["background:#EFF6FF;font-weight:700" for _ in row]
                return ["color:#9CA3AF" for _ in row]
            if row["Modelo"] == melhor["Modelo"]:
                return ["background:#EFF6FF;font-weight:600" for _ in row]
            return ["" for _ in row]

        st.dataframe(df_show.style.apply(hl, axis=1), hide_index=True, use_container_width=True)
        label_sub = sel if sel else melhor["Modelo"]
        st.markdown(f"<div style='font-size:.75rem;color:#6B7280;margin-top:6px'>Melhor por F1-Macro: <b style='color:#1B2B6B'>{melhor['Modelo']}</b></div>", unsafe_allow_html=True)

    with col_g:
        st.markdown("**Precision vs Recall por Modelo**")
        st.caption("Precision: quando classifica, quão frequentemente acerta · Recall: de todas as notícias de uma categoria, quantas encontrou")

        df_pr = pd.melt(
            df[["Modelo", "Precision", "Recall"]],
            id_vars="Modelo", value_vars=["Precision", "Recall"],
            var_name="Métrica", value_name="Valor"
        )

        fig_pr = go.Figure()
        for metrica, cor_fixo in [("Precision", "#2563EB"), ("Recall", "#0891B2")]:
            sub = df_pr[df_pr["Métrica"] == metrica].sort_values("Valor")
            if sel is None:
                # todos com cor fixa por métrica
                cores_pr = [cor_fixo] * len(sub)
            else:
                # modelo selecionado com cor fixa, outros cinzentos
                cores_pr = [cor_fixo if m == sel else "#D1D5DB" for m in sub["Modelo"]]
            fig_pr.add_trace(go.Bar(
                x=sub["Valor"], y=sub["Modelo"],
                orientation="h",
                name=metrica,
                marker_color=cores_pr,
                marker_line_width=0,
                text=sub["Valor"].map("{:.1f}%".format),
                textposition="outside",
                textfont_size=10,
                opacity=1.0,
            ))

        fig_pr.update_layout(
            **PLOTLY, height=300, barmode="group",
            margin=dict(l=0, r=55, t=10, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", ticksuffix="%",
                       zeroline=False, range=[0, 105]),
            yaxis=dict(showgrid=False, categoryorder="total ascending"),
            legend=dict(orientation="h", y=1.12, font_size=11),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # ── Ganho incremental por abordagem ──
    st.markdown("**Progressão de F1-Macro**")

    ordem_ganho = ["LLM 1B", "LLM 3B", "LLM 8B", "LLM Mistral", "TF-IDF", "Embeddings E5-Large", "Embeddings BGE-M3", "Embeddings Paraphrase"]
    df_ganho = df.set_index("Modelo").reindex(ordem_ganho).reset_index()
    df_ganho = df_ganho.dropna(subset=["F1-Macro"])

    fig_ganho = go.Figure()
    cores_ganho = [cor_modelo(m) for m in df_ganho["Modelo"]]
    sizes_ganho = [18 if (sel is None or m == sel) else 10 for m in df_ganho["Modelo"]]

    fig_ganho.add_trace(go.Scatter(
        x=df_ganho["Modelo"], y=df_ganho["F1-Macro"],
        mode="lines+markers+text",
        line=dict(color="#CBD5E1", width=2, dash="dot"),
        marker=dict(size=sizes_ganho, color=cores_ganho, line=dict(color="#fff", width=2)),
        text=df_ganho["F1-Macro"].map("{:.1f}%".format),
        textposition="top center",
        textfont=dict(size=12, color="#1F2937"),
        showlegend=False,
    ))
    fig_ganho.add_trace(go.Scatter(
        x=df_ganho["Modelo"], y=df_ganho["F1-Macro"],
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_ganho.update_layout(
        **PLOTLY, height=280,
        margin=dict(l=0, r=20, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", ticksuffix="%",
                   zeroline=False, range=[0, 105]),
    )
    st.plotly_chart(fig_ganho, use_container_width=True)

    st.divider()

    # ── Radar ──
    st.markdown("**Radar — Visão Global das Métricas**")
    dims = ["Accuracy", "Precision", "Recall", "F1-Macro"]
    fig_r = go.Figure()
    for _, row in df.iterrows():
        vals = [row[d] for d in dims] + [row[dims[0]]]
        cor = CORES.get(row["Modelo"], "#94A3B8")
        opa = opacidade_modelo(row["Modelo"])
        lw  = 3.5 if (sel == row["Modelo"]) else 2.0
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=dims + [dims[0]],
            name=row["Modelo"],
            line=dict(color=cor, width=lw),
            fill="toself", fillcolor=cor, opacity=opa,
        ))
    fig_r.update_layout(
        **PLOTLY, height=400,
        polar=dict(
            bgcolor="#FAFBFF",
            radialaxis=dict(range=[0, 100], ticksuffix="%",
                            tickfont_size=9, gridcolor="#E5E7EB"),
            angularaxis=dict(gridcolor="#E5E7EB", tickfont_size=12),
        ),
        legend=dict(orientation="h", y=-0.12, font_size=12),
        margin=dict(l=20, r=20, t=10, b=60),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    # ── Histograma — distribuição de F1 por categoria ──
    st.markdown("**Distribuição de F1 por Categoria**")
    st.caption("Quantas categorias cada modelo acerta bem (F1 alto) vs mal (F1 baixo)")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    hist_rows = []
    for nome, dfm in met_cat.items():
        counts = pd.cut(dfm["f1"], bins=bins, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
        for lbl, cnt in counts.items():
            hist_rows.append({"Modelo": nome, "F1": lbl, "Categorias": cnt})
    df_hist = pd.DataFrame(hist_rows)

    fig_hist = go.Figure()
    for nome in MODELOS:
        sub = df_hist[df_hist["Modelo"] == nome]
        fig_hist.add_trace(go.Bar(
            x=sub["F1"], y=sub["Categorias"],
            name=nome,
            marker_color=cor_modelo(nome),
            marker_line_width=0,
            opacity=opacidade_modelo(nome),
            text=sub["Categorias"],
            textposition="outside",
        ))
    fig_hist.update_layout(
        **PLOTLY, height=360, barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, title="Intervalo de F1-score"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Nº de categorias"),
        legend=dict(orientation="h", y=1.12, font_size=11),
        bargap=0.2, bargroupgap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Desconhecidas por modelo ──
    st.markdown("**Notícias classificadas como «desconhecido» por modelo**")
    desc_cols = st.columns(len(MODELOS))
    for i, (nome, col) in enumerate(MODELOS.items()):
        n_desc = int((av[col] == "desconhecido").sum())
        cor_d = "#B91C1B" if n_desc > 10 else "#B45309" if n_desc > 0 else "#15803D"
        with desc_cols[i]:
            cor_nome = CORES.get(nome, "#94A3B8")
            st.markdown(
                f"<div style='text-align:center;background:#fff;border:1px solid #E5E7EB;"
                f"border-radius:10px;padding:10px 6px'>"
                f"<div style='font-size:.6rem;font-weight:700;color:{cor_nome};"
                f"text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px'>{nome}</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:{cor_d};"
                f"font-family:IBM Plex Mono,monospace'>{n_desc}</div>"
                f"<div style='font-size:.6rem;color:#9CA3AF'>desconhecidas</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Heatmap F1 por categoria ──
    st.markdown("**Heatmap F1-Score por Categoria e Modelo**")
    dfs = []
    for nome, dfm in met_cat.items():
        tmp = dfm[["categoria", "f1"]].copy(); tmp["modelo"] = nome; dfs.append(tmp)
    cols_order = [sel] if sel else list(MODELOS)
    df_wide = pd.concat(dfs).pivot(index="categoria", columns="modelo", values="f1")[cols_order]
    fig_hm = px.imshow(df_wide, text_auto=".2f",
                       color_continuous_scale=["#FEE2E2", "#FEF3C7", "#D1FAE5"],
                       zmin=0, zmax=1, aspect="auto")
    fig_hm.update_layout(**PLOTLY, height=620 if not sel else 420,
                         margin=dict(l=0, r=0, t=10, b=0),
                         xaxis=dict(side="top"),
                         coloraxis_colorbar=dict(len=.5, thickness=10, tickformat=".0%"))
    fig_hm.update_traces(textfont_size=10)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # ── Matriz de Confusão ──
    st.markdown("**Matriz de Confusão**")
    st.markdown("""
    <div style='font-size:.8rem;color:#374151;background:#F8FAFC;border:1px solid #E5E7EB;border-radius:8px;padding:10px 14px;margin-bottom:10px'>
    📌 <b>Como ler:</b> cada célula mostra quantas notícias da <b>categoria real</b> (linha) foram classificadas como <b>categoria predita</b> (coluna).<br>
    A <b>diagonal</b> (azul escuro) são os <b>acertos</b> — o modelo previu corretamente.
    Os valores <b>fora da diagonal</b> são <b>erros</b> — o modelo confundiu uma categoria com outra.
    </div>
    """, unsafe_allow_html=True)

    modelo_cm = sel if sel else "Embeddings Paraphrase"
    if not sel:
        modelo_cm = st.selectbox("Modelo para matriz de confusão", list(MODELOS.keys()),
                                  index=list(MODELOS.keys()).index("Embeddings Paraphrase"))

    from sklearn.metrics import confusion_matrix as sk_cm
    col_cm = MODELOS[modelo_cm]
    categorias_ord = sorted(av["Categoria_humana"].unique())
    y_true_cm = av["Categoria_humana"].tolist()
    y_pred_cm = av[col_cm].tolist()
    cm = sk_cm(y_true_cm, y_pred_cm, labels=categorias_ord)
    df_cm = pd.DataFrame(cm, index=categorias_ord, columns=categorias_ord)

    fig_cm = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale=["#F9FAFB", "#BFDBFE", "#1D4ED8"],
        zmin=0, zmax=10, aspect="auto",
        labels=dict(x="Predito", y="Real", color="Artigos"),
    )
    fig_cm.update_layout(
        **PLOTLY, height=750,
        margin=dict(l=0, r=0, t=160, b=0),
        xaxis=dict(side="top", tickangle=-40, tickfont_size=10),
        yaxis=dict(tickfont_size=10),
        coloraxis_colorbar=dict(len=.4, thickness=10),
    )
    fig_cm.update_traces(textfont_size=10)
    st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — EXPLORADOR DE ARTIGOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Explorador de Notícias":
    st.title("Explorador de Notícias")
    st.caption("Todos os 240 artigos avaliados — filtra, explora e percebe onde cada modelo erra")

    # ── Filtros ──
    f1, f2, f3, f4 = st.columns([1.3, 2, 1.1, 1.1], gap="medium")
    with f1:
        modelo_sel = st.selectbox("Modelo", list(MODELOS.keys()))
    with f2:
        cats = sorted(av["Categoria_humana"].unique())
        cat_filtro = st.multiselect("Categoria", cats, placeholder="Todas")
    with f3:
        resultado = st.radio("Resultado", ["Todos", "Corretos", "Errados"], horizontal=True)
    with f4:
        ordenar = st.radio("Ordenar por", ["Título", "Score baixo", "Score alto"], horizontal=True)

    # ── Filtrar dados ──
    col_pred = MODELOS[modelo_sel]
    df_v = av.copy()
    df_v["correto"] = df_v["Categoria_humana"] == df_v[col_pred]
    if cat_filtro:
        df_v = df_v[df_v["Categoria_humana"].isin(cat_filtro)]
    if resultado == "Corretos":
        df_v = df_v[df_v["correto"]]
    elif resultado == "Errados":
        df_v = df_v[~df_v["correto"]]

    preds_m = preds[modelo_sel]

    # Attach confidence for sorting
    df_v = df_v.copy()
    def get_conf(art_id):
        info = preds_m.get(str(art_id), {})
        if info.get("confianca") is not None:
            return float(info["confianca"])
        if info.get("probs"):
            return max(info["probs"].values()) * 100
        return 100.0

    df_v["_conf"] = df_v["id"].map(get_conf)

    if ordenar == "Score baixo":
        df_v = df_v.sort_values("_conf", ascending=True)
    elif ordenar == "Score alto":
        df_v = df_v.sort_values("_conf", ascending=False)
    else:
        df_v = df_v.sort_values("titulo", ascending=True)

    total = len(df_v)
    corretos = int(df_v["correto"].sum())
    pct = corretos / total * 100 if total else 0
    cor_pct = "#15803D" if pct >= 80 else "#B45309" if pct >= 60 else "#B91C1B"

    st.markdown(
        f"<div class='stats'><b>{corretos}</b> de <b>{total}</b> artigos corretos — "
        f"<b style='color:{cor_pct}'>{pct:.1f}%</b> de acerto com <b>{modelo_sel}</b></div>",
        unsafe_allow_html=True,
    )

    # ── Lista de artigos ──
    for _, row in df_v.iterrows():
        art_id   = str(row["id"])
        pred_info = preds_m.get(art_id, {})
        correto  = row["correto"]
        cat_real = row["Categoria_humana"]
        cat_pred = pred_info.get("categoria", row[col_pred])
        conf     = pred_info.get("confianca")
        probs    = pred_info.get("probs", {})

        titulo = str(row["titulo"])
        titulo_curto = titulo[:85] + ("…" if len(titulo) > 85 else "")
        badge = "✓ Correto" if correto else "✗ Errado"
        badge_style = "color:#065F46;background:#D1FAE5" if correto else "color:#991B1B;background:#FEE2E2"

        with st.expander(titulo_curto):
            corpo = gold_text.get(art_id, "")
            if corpo:
                with st.expander("Ver texto completo"):
                    st.markdown(
                        f"<div style='font-size:.82rem;color:#374151;line-height:1.6'>{corpo[:3000]}{'…' if len(corpo)>3000 else ''}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            top_cols = st.columns([2, 3])

            with top_cols[0]:
                st.markdown(
                    f"<span style='{badge_style};padding:2px 10px;border-radius:20px;"
                    f"font-size:.72rem;font-weight:700;font-family:IBM Plex Mono,monospace'>{badge}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='font-size:.75rem;color:#6B7280;margin-bottom:3px'>Categoria real</div>"
                    f"<span class='b b-cat'>{cat_real}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:.75rem;color:#6B7280;margin-bottom:3px'>Predição do modelo</div>"
                    f"<span class='b b-alt'>{cat_pred}</span>",
                    unsafe_allow_html=True,
                )

                if conf is not None:
                    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                    conf_f = float(conf)
                    conf_color = "#15803D" if conf_f >= 80 else "#B45309" if conf_f >= 50 else "#B91C1B"
                    st.markdown(
                        f"<div style='font-size:.75rem;color:#6B7280;margin-bottom:4px'>Confiança</div>"
                        f"<span style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;"
                        f"font-weight:700;color:{conf_color}'>{conf_f:.0f}%</span>",
                        unsafe_allow_html=True,
                    )

            with top_cols[1]:
                if probs:
                    st.markdown(
                        "<div style='font-size:.75rem;color:#6B7280;margin-bottom:8px'>"
                        "Top 3 probabilidades <span style='font-size:.65rem;color:#9CA3AF'>"
                        "(azul = categoria predita)</span></div>",
                        unsafe_allow_html=True,
                    )
                    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    prob_bars({k: v for k, v in top3}, cat_real, pred_cat=cat_pred, top=3)
                elif not correto:
                    st.markdown(
                        "<div style='font-size:.75rem;color:#9CA3AF;font-style:italic'>"
                        "Probabilidades não disponíveis para este modelo</div>",
                        unsafe_allow_html=True,
                    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — NOTÍCIAS NOVAS (SILVER)
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Notícias Novas":
    st.title("Notícias Novas")
    st.caption("50 notícias do silver não vistas durante a avaliação — sem rótulo humano, compara o que cada modelo prevê")

    df_s, preds_s = load_silver()

    if df_s is None:
        st.error("Ficheiro preds_silver.json não encontrado. Corre primeiro: uv run python src/predict_silver.py")
        st.stop()

    MODELOS_S = ["TF-IDF", "Embeddings Paraphrase", "Embeddings E5-Large", "Embeddings BGE-M3", "LLM 1B", "LLM 3B", "LLM 8B", "LLM Mistral"]

    # ── Concordância global ──
    total = len(df_s)
    concordam_todos = 0
    concordam_4     = 0
    for _, row in df_s.iterrows():
        nid  = str(row["id"])
        cats = [preds_s[m][nid]["categoria"] for m in MODELOS_S if nid in preds_s.get(m, {})]
        if len(set(cats)) == 1:
            concordam_todos += 1
        elif len(set(cats)) <= 2:
            concordam_4 += 1

    c1, c2, c3, c_sp = st.columns([1, 1, 1, 1])
    with c1:
        kpi_card("Notícias", str(total), "amostra do silver", best=False)
    with c2:
        kpi_card("Todos concordam", f"{concordam_todos}/{total}",
                 f"{concordam_todos/total*100:.0f}% das notícias", best=concordam_todos > 0)
    with c3:
        kpi_card("≤ 2 categorias diferentes", f"{concordam_4}/{total}",
                 "modelos próximos", best=False)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Distribuição de categorias preditas por modelo ──
    st.markdown("**Distribuição de Categorias Preditas**")
    st.caption("O que cada modelo acha que são estas 50 notícias")

    dist_rows = []
    for modelo in MODELOS_S:
        for nid, info in preds_s.get(modelo, {}).items():
            dist_rows.append({"Modelo": modelo, "Categoria": info["categoria"]})
    df_dist = pd.DataFrame(dist_rows)
    df_dist_count = df_dist.groupby(["Modelo", "Categoria"]).size().reset_index(name="n")

    fig_dist = px.bar(df_dist_count, x="Categoria", y="n", color="Modelo",
                      barmode="group", color_discrete_map=CORES,
                      labels={"n": "Nº de notícias", "Categoria": ""})
    fig_dist.update_layout(**PLOTLY, height=350,
                           margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(tickangle=-35),
                           legend=dict(orientation="h", y=1.12, font_size=11),
                           bargap=0.2)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── Filtro ──
    st.markdown("**Explorar Notícias Individualmente**")
    fc1, fc2 = st.columns([2, 2], gap="medium")
    with fc1:
        ord_s = st.radio("Ordenar por", ["Divergência", "Consenso", "Título"], horizontal=True)
    with fc2:
        cat_filtro_s = st.multiselect("Filtrar por categoria predita (qualquer modelo)",
                                       sorted(df_dist["Categoria"].unique()), placeholder="Todas")

    # ── Ordenar ──
    rows_list = []
    for _, row in df_s.iterrows():
        nid  = str(row["id"])
        cats = [preds_s[m][nid]["categoria"] for m in MODELOS_S if nid in preds_s.get(m, {})]
        n_diferentes = len(set(cats))
        rows_list.append({"idx": _, "nid": nid, "n_dif": n_diferentes, "cats": cats})

    titulos_map = dict(zip(df_s["id"].astype(str), df_s["titulo"]))
    df_ord = pd.DataFrame(rows_list)
    df_ord["titulo_s"] = df_ord["nid"].map(titulos_map)

    if ord_s == "Divergência":
        df_ord = df_ord.sort_values(["n_dif", "titulo_s"], ascending=[False, True])
    elif ord_s == "Consenso":
        df_ord = df_ord.sort_values(["n_dif", "titulo_s"], ascending=[True, True])
    elif ord_s == "Título":
        df_ord = df_ord.sort_values("titulo_s")

    if cat_filtro_s:
        df_ord = df_ord[df_ord["cats"].apply(lambda cs: any(c in cat_filtro_s for c in cs))]

    # ── Lista ──
    for _, orow in df_ord.iterrows():
        nid       = orow["nid"]
        art_row   = df_s[df_s["id"].astype(str) == nid].iloc[0]
        titulo    = str(art_row["titulo"])
        corpo     = str(art_row["noticia_norm"])
        cats_pred = orow["cats"]
        n_dif     = orow["n_dif"]

        titulo_curto = titulo[:85] + ("…" if len(titulo) > 85 else "")

        if n_dif == 1:
            badge_txt   = "✓ Consenso"
            badge_style = "color:#065F46;background:#D1FAE5"
        elif n_dif == 2:
            badge_txt   = "~ Quase consenso"
            badge_style = "color:#92400E;background:#FEF3C7"
        else:
            badge_txt   = f"✗ Divergência ({n_dif} categorias)"
            badge_style = "color:#991B1B;background:#FEE2E2"

        with st.expander(titulo_curto):
            st.markdown(
                f"<span style='{badge_style};padding:2px 10px;border-radius:20px;"
                f"font-size:.72rem;font-weight:700;font-family:IBM Plex Mono,monospace'>{badge_txt}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # Texto completo
            with st.expander("Ver texto completo"):
                st.markdown(f"<div style='font-size:.82rem;color:#374151;line-height:1.6'>{corpo[:3000]}{'…' if len(corpo)>3000 else ''}</div>",
                            unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Predições de cada modelo lado a lado
            cols_m = st.columns(len(MODELOS_S))
            for ci, modelo in enumerate(MODELOS_S):
                info  = preds_s.get(modelo, {}).get(nid, {})
                cat   = info.get("categoria", "—")
                probs = info.get("probs", {})
                conf  = info.get("confianca")
                cor   = CORES.get(modelo, "#94A3B8")

                with cols_m[ci]:
                    st.markdown(
                        f"<div style='font-size:.65rem;font-weight:700;color:{cor};"
                        f"text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px'>{modelo}</div>"
                        f"<span class='b b-cat'>{cat}</span>",
                        unsafe_allow_html=True,
                    )
                    if conf is not None:
                        conf_color = "#15803D" if conf >= 80 else "#B45309" if conf >= 50 else "#B91C1B"
                        st.markdown(
                            f"<div style='font-size:.7rem;color:{conf_color};"
                            f"font-family:IBM Plex Mono,monospace;margin-top:4px'>conf: {conf:.0f}%</div>",
                            unsafe_allow_html=True,
                        )
                    if probs:
                        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                        prob_bars({k: v for k, v in top3}, cat, pred_cat=cat, top=3)
