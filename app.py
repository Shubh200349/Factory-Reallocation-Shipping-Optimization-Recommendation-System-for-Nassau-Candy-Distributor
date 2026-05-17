# ============================================================
#  app.py  —  Nassau Candy: Streamlit Dashboard
#  HOW TO RUN:  streamlit run app.py
#  Run 02_model.py FIRST to generate the model files.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go

# ── Page config (must be FIRST Streamlit call) ────────────────
st.set_page_config(
    page_title="Nassau Candy Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  THEME STATE
# ══════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

IS_DARK = st.session_state.theme == "dark"

# ── Color tokens ──────────────────────────────────────────────
if IS_DARK:
    BG_BASE     = "#0e1a27"
    BG_SURFACE  = "#152030"
    BG_ELEVATED = "#1b2a3d"
    BG_INPUT    = "#1b2a3d"
    BORDER      = "rgba(255,255,255,0.08)"
    BORDER_HI   = "rgba(255,255,255,0.18)"
    TEXT1       = "#eee9e0"
    TEXT2       = "#8fa5bb"
    TEXT3       = "#4d6880"
    GOLD        = "#f0b429"
    TEAL        = "#2dd4bf"
    CORAL       = "#f87171"
    BLUE        = "#60a5fa"
    VIOLET      = "#c084fc"
    HERO_GRAD   = "linear-gradient(135deg,#1b2a3d 0%,#0e1a27 55%,#152030 100%)"
    SHADOW      = "0 4px 28px rgba(0,0,0,0.5)"
    GLOW        = "rgba(240,180,41,0.09)"
    MAP_LAND    = "#1a2d40"
    MAP_OCEAN   = "#0e1a27"
    HEAT_CS     = [[0,"#0e1a27"],[0.5,"#1e4976"],[1,"#f0b429"]]
    PLT_FONT    = "#8fa5bb"
    GRID        = "rgba(255,255,255,0.04)"
    AXIS_LINE   = "rgba(255,255,255,0.08)"
    # multiselect tag colours in dark
    TAG_BG      = "#253d55"
    TAG_TEXT    = "#eee9e0"
    TAG_BORDER  = "rgba(255,255,255,0.15)"
    # input box
    INPUT_BG    = "#1b2a3d"
    INPUT_TEXT  = "#eee9e0"
    INPUT_BORDER= "rgba(255,255,255,0.12)"
    # scrollbar / dropdown
    DROP_BG     = "#1b2a3d"
    DROP_HOVER  = "#253d55"
    BTN_BG      = "#1b2a3d"
    BTN_BORDER  = BORDER_HI
    BTN_TEXT    = "#eee9e0"
    BTN_ICON    = "☀️"
    BTN_LABEL   = "Switch to Light Mode"
    SLIDER_COLOR= GOLD
else:
    BG_BASE     = "#f0ebe1"
    BG_SURFACE  = "#ffffff"
    BG_ELEVATED = "#faf7f2"
    BG_INPUT    = "#ffffff"
    BORDER      = "rgba(0,0,0,0.08)"
    BORDER_HI   = "rgba(0,0,0,0.18)"
    TEXT1       = "#1a1f2e"
    TEXT2       = "#4a5568"
    TEXT3       = "#8a9ab0"
    GOLD        = "#b45309"
    TEAL        = "#0d9488"
    CORAL       = "#dc2626"
    BLUE        = "#2563eb"
    VIOLET      = "#7c3aed"
    HERO_GRAD   = "linear-gradient(135deg,#fff8ef 0%,#f0ebe1 55%,#fdf9f2 100%)"
    SHADOW      = "0 4px 24px rgba(0,0,0,0.09)"
    GLOW        = "rgba(180,83,9,0.06)"
    MAP_LAND    = "#cde3f5"
    MAP_OCEAN   = "#a8c8e8"
    HEAT_CS     = [[0,"#dbeafe"],[0.5,"#3b82f6"],[1,"#b45309"]]
    PLT_FONT    = "#4a5568"
    GRID        = "rgba(0,0,0,0.05)"
    AXIS_LINE   = "rgba(0,0,0,0.09)"
    # multiselect tag colours in light  ← KEY FIX
    TAG_BG      = "#e8f0fb"
    TAG_TEXT    = "#1a3a6b"
    TAG_BORDER  = "rgba(37,99,235,0.25)"
    # input box
    INPUT_BG    = "#ffffff"
    INPUT_TEXT  = "#1a1f2e"
    INPUT_BORDER= "rgba(0,0,0,0.15)"
    # dropdown
    DROP_BG     = "#ffffff"
    DROP_HOVER  = "#f0ebe1"
    BTN_BG      = "#1a1f2e"
    BTN_BORDER  = "#1a1f2e"
    BTN_TEXT    = "#ffffff"
    BTN_ICON    = "🌙"
    BTN_LABEL   = "Switch to Dark Mode"
    SLIDER_COLOR= GOLD

# ══════════════════════════════════════════════════════════════
#  CSS — every value comes from Python variables above
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {BG_BASE} !important;
    color: {TEXT1} !important;
}}
.stApp {{ background-color: {BG_BASE} !important; }}
.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 2rem !important;
    max-width: 1180px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {BG_SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: {BG_SURFACE} !important;
}}

/* ── ALL sidebar text — override dark defaults ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {TEXT1} !important;
}}

/* ── Input labels everywhere ── */
label, .stSelectbox label, .stMultiSelect label,
[data-testid="stWidgetLabel"] p {{
    color: {TEXT2} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}}

/* ── Selectbox / dropdown box ── */
[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    border: 1px solid {INPUT_BORDER} !important;
    border-radius: 8px !important;
    color: {INPUT_TEXT} !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {{
    color: {INPUT_TEXT} !important;
    background-color: transparent !important;
}}

/* ── Dropdown menu list ── */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {{
    background-color: {DROP_BG} !important;
    border: 1px solid {INPUT_BORDER} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] li span {{
    color: {INPUT_TEXT} !important;
    background-color: {DROP_BG} !important;
}}
div[data-baseweb="menu"] li:hover {{
    background-color: {DROP_HOVER} !important;
}}

/* ── Multiselect box ── */
div[data-baseweb="input"],
div[data-baseweb="base-input"] {{
    background-color: {INPUT_BG} !important;
    color: {INPUT_TEXT} !important;
}}
[data-testid="stMultiSelect"] > div > div {{
    background-color: {INPUT_BG} !important;
    border: 1px solid {INPUT_BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stMultiSelect"] input {{
    color: {INPUT_TEXT} !important;
    background-color: transparent !important;
}}

/* ── Multiselect TAGS (the coloured pills) — KEY FIX ── */
span[data-baseweb="tag"],
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background-color: {TAG_BG} !important;
    border: 1px solid {TAG_BORDER} !important;
    border-radius: 6px !important;
}}
span[data-baseweb="tag"] span,
span[data-baseweb="tag"] > span {{
    color: {TAG_TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}}
/* tag close (×) button */
span[data-baseweb="tag"] button,
span[data-baseweb="tag"] [role="presentation"] {{
    color: {TAG_TEXT} !important;
    opacity: 0.7;
}}
span[data-baseweb="tag"] button:hover {{ opacity: 1; }}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {{
    background: {SLIDER_COLOR} !important;
}}

/* ── Theme toggle button ── */
[data-testid="stSidebar"] .stButton > button {{
    width: 100% !important;
    background: {BTN_BG} !important;
    color: {BTN_TEXT} !important;
    border: 1px solid {BTN_BORDER} !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: opacity .2s !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    opacity: 0.82 !important;
    border-color: {GOLD} !important;
}}
[data-testid="stSidebar"] .stButton > button:focus {{
    box-shadow: none !important;
    outline: none !important;
}}

/* ── Hero ── */
.hero-wrap {{
    background: {HERO_GRAD};
    border: 1px solid {BORDER_HI};
    border-radius: 16px;
    padding: 1.8rem 2.2rem 1.4rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: {SHADOW};
}}
.hero-wrap::before {{
    content: '';
    position: absolute; top: -50px; right: -50px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, {GLOW} 0%, transparent 70%);
    pointer-events: none;
}}
.hero-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
    color: {GOLD} !important; margin-bottom: 7px;
}}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.85rem; font-weight: 700;
    color: {TEXT1} !important; line-height: 1.2; margin-bottom: 5px;
}}
.hero-sub {{ font-size: .86rem; color: {TEXT2} !important; font-weight: 300; }}

/* ── KPI cards ── */
.kpi-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 13px;
    padding: 1.05rem 1.1rem 0.9rem;
    box-shadow: {SHADOW};
    transition: border-color .2s, transform .2s;
    position: relative; overflow: hidden; margin-bottom: 4px;
}}
.kpi-card:hover {{ border-color: {BORDER_HI}; transform: translateY(-2px); }}
.kpi-card::after {{
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2.5px;
    background: linear-gradient(90deg, {GOLD}, {TEAL}); opacity: .65;
}}
.kpi-label {{
    font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500;
    letter-spacing: .09em; text-transform: uppercase; color: {TEXT3} !important; margin-bottom: 5px;
}}
.kpi-value {{
    font-family: 'Playfair Display', serif; font-size: 1.62rem; font-weight: 700;
    color: {TEXT1} !important; line-height: 1.1;
}}
.kpi-delta {{ font-size: 11px; margin-top: 3px; color: {TEAL} !important; }}

/* ── Section title ── */
.sec-title {{
    font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 600;
    color: {TEXT1} !important; margin: 1.3rem 0 .55rem;
    display: flex; align-items: center; gap: 8px;
}}
.sec-title::after {{ content: ''; flex: 1; height: 1px; background: {BORDER}; margin-left: 6px; }}

/* ── Metric widget ── */
[data-testid="metric-container"] {{
    background: {BG_SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: .9rem 1rem !important;
}}
[data-testid="metric-container"] label {{
    color: {TEXT3} !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: .07em !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important; color: {TEXT1} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 12px !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: {TEXT2} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: {GOLD} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: {BORDER} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: {GOLD} !important; }}

/* ── Recommendation card ── */
.rec-card {{
    background: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-left: 3px solid {GOLD};
    border-radius: 11px;
    padding: .95rem 1.15rem; margin-bottom: .65rem;
    box-shadow: {SHADOW};
}}
.rec-rank    {{ font-family:'DM Mono',monospace; font-size:10px; color:{GOLD} !important; text-transform:uppercase; letter-spacing:.1em; }}
.rec-product {{ font-family:'Playfair Display',serif; font-size:.98rem; color:{TEXT1} !important; margin:2px 0 6px; }}
.rec-meta    {{ font-size:12px; color:{TEXT2} !important; }}
.rec-badge   {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:10px; font-weight:500; margin-right:5px; }}
.badge-h  {{ background:rgba(45,212,191,.13);  color:{TEAL}  !important; border:1px solid rgba(45,212,191,.3); }}
.badge-m  {{ background:rgba(240,180,41,.13);  color:{GOLD}  !important; border:1px solid rgba(240,180,41,.3); }}
.badge-l  {{ background:rgba(248,113,113,.13); color:{CORAL} !important; border:1px solid rgba(248,113,113,.3); }}
.badge-dv {{ background:rgba(96,165,250,.13);  color:{BLUE}  !important; border:1px solid rgba(96,165,250,.3); }}

/* ── Sidebar labels ── */
.sb-name {{ font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:{TEXT1} !important; }}
.sb-sub  {{ font-family:'DM Mono',monospace; font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:{TEXT3} !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important; overflow: hidden !important;
    border: 1px solid {BORDER} !important;
}}
/* dataframe cell text */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {{
    color: {TEXT1} !important;
    background-color: {BG_SURFACE} !important;
}}

/* ── Alerts ── */
.stAlert {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
    background: {BG_ELEVATED} !important;
}}

/* ── General text ── */
p, h1, h2, h3, h4 {{ color: {TEXT1} !important; }}
hr {{ border: none !important; border-top: 1px solid {BORDER} !important; margin: 1.1rem 0 !important; }}

/* hide footer */
#MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly layout ─────────────────────────────────────────────
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color=PLT_FONT, size=12),
    margin=dict(t=20, b=32, l=8, r=8),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    colorway=[GOLD, TEAL, BLUE, VIOLET, CORAL, '#fbbf24'],
    xaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE, tickfont=dict(size=11, color=PLT_FONT)),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS_LINE, tickfont=dict(size=11, color=PLT_FONT)),
)
def aplt(fig, h=290):
    fig.update_layout(**PL, height=h)
    return fig

# ── Constants ─────────────────────────────────────────────────
PRODUCT_FACTORY = {
    'Wonka Bar - Nutty Crunch Surprise':    "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows':            "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious':       "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate':           "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel':    "Wicked Choccy's",
    'Laffy Taffy':                          'Sugar Shack',
    'SweeTARTS':                            'Sugar Shack',
    'Nerds':                                'Sugar Shack',
    'Fun Dip':                              'Sugar Shack',
    'Fizzy Lifting Drinks':                 'Sugar Shack',
    'Everlasting Gobstopper':               'Secret Factory',
    'Hair Toffee':                          'The Other Factory',
    'Lickable Wallpaper':                   'Secret Factory',
    'Wonka Gum':                            'Secret Factory',
    'Kazookles':                            'The Other Factory',
}
ALL_FACTORIES = ["Lot's O' Nuts","Wicked Choccy's","Sugar Shack","Secret Factory","The Other Factory"]
FACTORY_COORDS = {
    "Lot's O' Nuts":    (32.881803,-111.768036),
    "Wicked Choccy's":  (32.076176, -81.088371),
    "Sugar Shack":      (48.110140, -96.138150),
    "Secret Factory":   (41.446333, -90.565487),
    "The Other Factory":(35.117500, -89.971107),
}

# ── Load data & model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return (joblib.load('models/lead_time_model.pkl'),
                joblib.load('models/label_encoders.pkl'),
                joblib.load('models/feature_names.pkl'))
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/processed_data.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/Nassau_Candy_Distributor.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
        df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
        df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days
        df['Factory']    = df['Product Name'].map(PRODUCT_FACTORY)
        df['Margin_Pct'] = df['Gross Profit'] / df['Sales'] * 100
        return df

model, encoders, feature_names = load_model()
df = load_data()

def safe_enc(enc, val):
    return enc.transform([val])[0] if val in list(enc.classes_) else 0

def predict(region, ship_mode, factory, division, sales, units, profit, cost):
    if not model: return None
    margin = (profit / sales * 100) if sales > 0 else 0
    row = {
        'Region_enc':    safe_enc(encoders['Region'],    region),
        'Ship Mode_enc': safe_enc(encoders['Ship Mode'], ship_mode),
        'Factory_enc':   safe_enc(encoders['Factory'],   factory),
        'Division_enc':  safe_enc(encoders['Division'],  division),
        'Sales': sales, 'Units': units,
        'Gross Profit': profit, 'Cost': cost, 'Margin_Pct': margin,
    }
    return round(float(model.predict(pd.DataFrame([row])[feature_names])[0]), 1)

def division_of(p):
    if 'Wonka' in p: return 'Chocolate'
    if p in ['Laffy Taffy','SweeTARTS','Nerds','Fun Dip','Everlasting Gobstopper','Hair Toffee']: return 'Sugar'
    return 'Other'

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:0.3rem 0 0.9rem;border-bottom:1px solid {BORDER};margin-bottom:0.9rem;">
        <div style="font-size:26px;margin-bottom:4px;">🍬</div>
        <div class="sb-name">Nassau Candy</div>
        <div class="sb-sub">Factory Intelligence</div>
    </div>
    <p class="sb-sub" style="margin-bottom:6px;">Appearance</p>
    """, unsafe_allow_html=True)

    if st.button(f"{BTN_ICON}  {BTN_LABEL}", key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown(f'<p style="font-size:10px;color:{TEXT3};margin-top:4px;margin-bottom:1rem;">Currently: <b style="color:{TEXT2};">{"Dark" if IS_DARK else "Light"}</b> mode</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="border-top:1px solid {BORDER};margin-bottom:0.9rem;"></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sb-sub" style="margin-bottom:8px;">Filter Dataset</p>', unsafe_allow_html=True)

    region_f   = st.multiselect("Region",    df['Region'].unique(),    default=list(df['Region'].unique()))
    division_f = st.multiselect("Division",  df['Division'].unique(),  default=list(df['Division'].unique()))
    ship_f     = st.multiselect("Ship Mode", df['Ship Mode'].unique(), default=list(df['Ship Mode'].unique()))

    st.markdown(f'<div style="border-top:1px solid {BORDER};margin:0.9rem 0;"></div>', unsafe_allow_html=True)
    st.success("✅ Model active") if model else st.error("⚠️ Run 02_model.py first")
    st.markdown(f'<p style="font-size:11px;color:{TEXT3};margin-top:.8rem;line-height:1.7;">10,194 orders · 15 products<br>5 factories · 4 regions</p>', unsafe_allow_html=True)

dff = df[df['Region'].isin(region_f) & df['Division'].isin(division_f) & df['Ship Mode'].isin(ship_f)]

# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">◆ Decision Intelligence Platform</div>
    <div class="hero-title">Nassau Candy — Factory Optimization</div>
    <div class="hero-sub">Predictive shipping intelligence &nbsp;·&nbsp; Factory reallocation &nbsp;·&nbsp; Scenario simulation engine</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview", "🔮  Factory Simulator", "⚡  What-If Analysis", "🏆  Recommendations"
])

# ──────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────
with tab1:
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [
        ("Total Orders",  f"{len(dff):,}",                                        "all time"),
        ("Total Sales",   f"${dff['Sales'].sum():,.0f}",                          "revenue"),
        ("Total Profit",  f"${dff['Gross Profit'].sum():,.0f}",                   "gross"),
        ("Avg Lead Time", f"{dff['Lead Time'].mean():.0f} days",                  "shipping"),
        ("Avg Margin",    f"{(dff['Gross Profit']/dff['Sales']*100).mean():.1f}%","profit"),
    ]
    for col,(lbl,val,dlt) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-delta">↗ {dlt}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec-title">Sales & Profit by Factory</div>', unsafe_allow_html=True)
        fkpi = dff.groupby('Factory').agg(Sales=('Sales','sum'), Profit=('Gross Profit','sum')).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Sales',  x=fkpi['Factory'], y=fkpi['Sales'],  marker_color=GOLD, marker_line_width=0))
        fig.add_trace(go.Bar(name='Profit', x=fkpi['Factory'], y=fkpi['Profit'], marker_color=TEAL, marker_line_width=0))
        aplt(fig.update_layout(barmode='group', xaxis_tickangle=-18), 290)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec-title">Lead Time Heatmap — Region × Ship Mode</div>', unsafe_allow_html=True)
        heat = dff.groupby(['Region','Ship Mode'])['Lead Time'].mean().round(1).unstack(fill_value=0)
        fig2 = px.imshow(heat, text_auto=True, aspect='auto', color_continuous_scale=HEAT_CS)
        fig2.update_coloraxes(showscale=False)
        aplt(fig2, 290)
        st.plotly_chart(fig2, use_container_width=True)

    c3,c4,c5 = st.columns(3)
    with c3:
        st.markdown('<div class="sec-title">Orders by Division</div>', unsafe_allow_html=True)
        dc = dff['Division'].value_counts().reset_index()
        fig3 = px.pie(dc, names='Division', values='count', hole=.55,
                      color_discrete_sequence=[GOLD, TEAL, BLUE])
        fig3.update_traces(textposition='outside', textfont_size=10)
        aplt(fig3.update_layout(showlegend=True,
             legend=dict(orientation='h',y=-0.18,x=.5,xanchor='center')), 250)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="sec-title">Lead Time by Ship Mode</div>', unsafe_allow_html=True)
        lt_s = dff.groupby('Ship Mode')['Lead Time'].mean().sort_values().reset_index()
        fig4 = px.bar(lt_s, x='Lead Time', y='Ship Mode', orientation='h',
                      color='Lead Time', color_continuous_scale=[[0,TEAL],[1,GOLD]])
        fig4.update_traces(marker_line_width=0)
        fig4.update_coloraxes(showscale=False)
        aplt(fig4.update_layout(showlegend=False), 250)
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        st.markdown('<div class="sec-title">Margin % by Division</div>', unsafe_allow_html=True)
        tmp = dff.copy(); tmp['Margin'] = tmp['Gross Profit']/tmp['Sales']*100
        md = tmp.groupby('Division')['Margin'].mean().reset_index()
        fig5 = px.bar(md, x='Division', y='Margin',
                      color='Margin', color_continuous_scale=[[0,BLUE],[1,VIOLET]])
        fig5.update_traces(marker_line_width=0)
        fig5.update_coloraxes(showscale=False)
        aplt(fig5.update_layout(showlegend=False, yaxis_title='Avg Margin %'), 250)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="sec-title">Factory Locations — United States</div>', unsafe_allow_html=True)
    map_df = pd.DataFrame([{
        'Factory':k,'Latitude':v[0],'Longitude':v[1],
        'Orders':int(dff[dff['Factory']==k].shape[0]),
        'Profit':float(dff[dff['Factory']==k]['Gross Profit'].sum()),
    } for k,v in FACTORY_COORDS.items()])
    fig_m = px.scatter_geo(map_df, lat='Latitude', lon='Longitude', text='Factory',
                           size='Orders', color='Profit', scope='usa',
                           color_continuous_scale=[[0,MAP_LAND],[0.5,BLUE],[1,GOLD]],
                           size_max=44,
                           hover_data={'Orders':True,'Profit':':,.0f','Latitude':False,'Longitude':False})
    fig_m.update_traces(textfont=dict(color=TEXT1, size=11))
    fig_m.update_geos(bgcolor='rgba(0,0,0,0)', landcolor=MAP_LAND, oceancolor=MAP_OCEAN,
                      lakecolor=MAP_OCEAN, showland=True, showocean=True, showlakes=True,
                      coastlinecolor=BORDER_HI, countrycolor=BORDER)
    fig_m.update_coloraxes(showscale=False)
    aplt(fig_m, 380)
    st.plotly_chart(fig_m, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  TAB 2 — FACTORY SIMULATOR
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">Factory Simulator</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT2};font-size:13px;margin-bottom:1rem;">Select any product + factory combination to get an instant predicted lead time.</p>', unsafe_allow_html=True)

    if not model:
        st.error("⚠️ Model not found. Please run **02_model.py** first, then restart the dashboard.")
    else:
        s1,s2 = st.columns([1,1], gap="large")
        with s1:
            st.markdown(f'<p class="sb-sub" style="margin-bottom:10px;">Configure Scenario</p>', unsafe_allow_html=True)
            sel_prod = st.selectbox("Product",           list(PRODUCT_FACTORY.keys()))
            sel_fac  = st.selectbox("Target Factory",    ALL_FACTORIES)
            sel_reg  = st.selectbox("Destination Region",df['Region'].unique().tolist())
            sel_ship = st.selectbox("Ship Mode",         df['Ship Mode'].unique().tolist())
            sel_u    = st.slider("Units", 1, 15, 5)
            sel_s    = st.slider("Sales ($)", 1.0, 260.0, 50.0, step=0.5)
            sel_p    = st.slider("Gross Profit ($)", 0.25, 130.0, min(round(sel_s*0.65,2),130.0), step=0.25)
            sel_c    = round(sel_s - sel_p, 2)
            sel_div  = division_of(sel_prod)

        with s2:
            cur_fac = PRODUCT_FACTORY.get(sel_prod, ALL_FACTORIES[0])
            cur_lt  = predict(sel_reg, sel_ship, cur_fac, sel_div, sel_s, sel_u, sel_p, sel_c)
            tgt_lt  = predict(sel_reg, sel_ship, sel_fac, sel_div, sel_s, sel_u, sel_p, sel_c)

            if cur_lt and tgt_lt:
                delta = tgt_lt - cur_lt
                dp    = (delta/cur_lt*100) if cur_lt else 0
                m1,m2 = st.columns(2)
                m1.metric("Current Factory",    cur_fac)
                m1.metric("Current Lead Time",  f"{cur_lt} days")
                m2.metric("Target Factory",     sel_fac)
                m2.metric("Predicted Lead Time",f"{tgt_lt} days",
                          delta=f"{delta:+.1f} days ({dp:+.1f}%)", delta_color="inverse")

                if delta < 0:   st.success(f"✅ Moving to **{sel_fac}** saves **{abs(delta):.1f} days**!")
                elif delta > 0: st.warning(f"⚠️ Moving to **{sel_fac}** adds **{delta:.1f} days** to lead time.")
                else:           st.info("ℹ️ No significant change for this configuration.")

                st.markdown('<div class="sec-title" style="margin-top:1rem;">All-Factory Comparison</div>', unsafe_allow_html=True)
                ap    = [{'Factory':f,'Lead Time':predict(sel_reg,sel_ship,f,sel_div,sel_s,sel_u,sel_p,sel_c)} for f in ALL_FACTORIES]
                ap_df = pd.DataFrame(ap).sort_values('Lead Time')
                min_lt = ap_df['Lead Time'].min()
                colors = [CORAL if r.Factory==cur_fac else TEAL if r['Lead Time']==min_lt else BLUE for _,r in ap_df.iterrows()]
                fig_b = go.Figure(go.Bar(
                    x=ap_df['Factory'], y=ap_df['Lead Time'],
                    marker_color=colors, marker_line_width=0,
                    text=ap_df['Lead Time'].round(1), textposition='outside',
                    textfont=dict(color=TEXT2, size=11),
                ))
                aplt(fig_b.update_layout(yaxis_title='Lead Time (days)'), 260)
                st.plotly_chart(fig_b, use_container_width=True)
                st.markdown(f'<p style="font-size:11px;color:{TEXT3};">🟩 Best &nbsp;·&nbsp; 🟥 Current &nbsp;·&nbsp; 🟦 Alternatives</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  TAB 3 — WHAT-IF
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">What-If Scenario Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT2};font-size:13px;margin-bottom:1rem;">Simulate optimal reassignment for every product under a chosen region & ship mode.</p>', unsafe_allow_html=True)

    if not model:
        st.error("⚠️ Model not found. Please run **02_model.py** first.")
    else:
        wa1,wa2 = st.columns(2)
        wa_reg  = wa1.selectbox("Region",    df['Region'].unique().tolist(), key='wa_r')
        wa_ship = wa2.selectbox("Ship Mode", df['Ship Mode'].unique().tolist(), key='wa_s')

        rows = []
        for prod,cfac in PRODUCT_FACTORY.items():
            div = division_of(prod)
            ar  = df[df['Product Name']==prod]
            if ar.empty: continue
            as_,ap_,ac_,au_ = ar['Sales'].mean(),ar['Gross Profit'].mean(),ar['Cost'].mean(),ar['Units'].mean()
            clt = predict(wa_reg,wa_ship,cfac,div,as_,au_,ap_,ac_)
            blt,bfac = clt,cfac
            for f in ALL_FACTORIES:
                lt = predict(wa_reg,wa_ship,f,div,as_,au_,ap_,ac_)
                if lt and lt < blt: blt,bfac = lt,f
            saving = clt-blt if clt and blt else 0
            rows.append({'Product':prod,'Division':div,'Current Factory':cfac,
                         'Current LT':round(clt,1) if clt else 0,
                         'Best Factory':bfac,'Best LT':round(blt,1) if blt else 0,
                         'Days Saved':round(saving,1),
                         'Action':'✅ Keep' if bfac==cfac else '🔄 Reassign'})

        rdf = pd.DataFrame(rows).sort_values('Days Saved',ascending=False)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Reassignments",    (rdf['Action']=='🔄 Reassign').sum())
        m2.metric("Total Days Saved", f"{rdf['Days Saved'].sum():.1f}")
        m3.metric("Avg Days Saved",   f"{rdf['Days Saved'].mean():.1f}")
        m4.metric("Products Analyzed", len(rdf))
        st.markdown("")
        st.dataframe(rdf, use_container_width=True, hide_index=True,
                     column_config={'Days Saved':st.column_config.NumberColumn(format="%.1f ⬇")})

        st.markdown('<div class="sec-title">Current vs Optimal Lead Time</div>', unsafe_allow_html=True)
        fig_wa = go.Figure()
        fig_wa.add_trace(go.Bar(name='Current LT', x=rdf['Product'], y=rdf['Current LT'], marker_color=CORAL, marker_line_width=0))
        fig_wa.add_trace(go.Bar(name='Best LT',    x=rdf['Product'], y=rdf['Best LT'],    marker_color=TEAL,  marker_line_width=0))
        aplt(fig_wa.update_layout(barmode='group', xaxis_tickangle=-28, yaxis_title='Lead Time (days)',
                                  legend=dict(orientation='h',y=1.06,x=0)), 340)
        st.plotly_chart(fig_wa, use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  TAB 4 — RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-title">Ranked Factory Reassignment Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT2};font-size:13px;margin-bottom:1rem;">Priority = lead time saving × profit margin. Focus on rank #1 first.</p>', unsafe_allow_html=True)

    if not model:
        st.error("⚠️ Model not found. Please run **02_model.py** first.")
    else:
        r1,r2 = st.columns(2)
        rec_reg  = r1.selectbox("Region",    df['Region'].unique().tolist(), key='rc_r')
        rec_ship = r2.selectbox("Ship Mode", df['Ship Mode'].unique().tolist(), key='rc_s')

        recs = []
        for prod,cfac in PRODUCT_FACTORY.items():
            div = division_of(prod)
            ar  = df[df['Product Name']==prod]
            if ar.empty: continue
            as_,ap_,ac_,au_ = ar['Sales'].mean(),ar['Gross Profit'].mean(),ar['Cost'].mean(),ar['Units'].mean()
            clt = predict(rec_reg,rec_ship,cfac,div,as_,au_,ap_,ac_)
            blt,bfac = clt,cfac
            for f in ALL_FACTORIES:
                lt = predict(rec_reg,rec_ship,f,div,as_,au_,ap_,ac_)
                if lt and lt < blt: blt,bfac = lt,f
            if bfac != cfac:
                saving = clt-blt; sp = (saving/clt*100) if clt else 0
                margin = ap_/as_*100 if as_>0 else 0
                conf   = 'High' if saving>50 else ('Medium' if saving>10 else 'Low')
                recs.append({'priority':saving*(margin/100),'product':prod,'division':div,
                             'from_fac':cfac,'to_fac':bfac,'saving':saving,
                             'sp':sp,'margin':margin,'conf':conf})

        recs_df = pd.DataFrame(recs).sort_values('priority',ascending=False).reset_index(drop=True) if recs else pd.DataFrame()

        if recs_df.empty:
            st.success("✅ All products are already optimally assigned for this combination!")
        else:
            s1,s2,s3 = st.columns(3)
            s1.metric("Reassignments",    len(recs_df))
            s2.metric("Total Days Saved", f"{recs_df['saving'].sum():.1f}")
            top = recs_df.iloc[0]['product']
            s3.metric("Top Priority", top[:24]+"…" if len(top)>24 else top)
            st.markdown("")

            for i,row in recs_df.iterrows():
                bc = {'High':'badge-h','Medium':'badge-m','Low':'badge-l'}.get(row['conf'],'badge-m')
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-rank">#{i+1} Recommendation</div>
                    <div class="rec-product">{row['product']}</div>
                    <div style="margin-bottom:7px;">
                        <span class="rec-badge {bc}">{row['conf']} confidence</span>
                        <span class="rec-badge badge-dv">{row['division']}</span>
                    </div>
                    <div class="rec-meta">
                        <b style="color:{TEXT1};">Move:</b> {row['from_fac']}
                        &nbsp;→&nbsp;<b style="color:{GOLD};">{row['to_fac']}</b>
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        <b style="color:{TEAL};">Save {row['saving']:.1f} days ({row['sp']:.1f}%)</b>
                        &nbsp;&nbsp;·&nbsp;&nbsp;Margin: {row['margin']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:1.5rem;">⚠️ Risk Panel</div>', unsafe_allow_html=True)
        rp1,rp2 = st.columns(2)
        with rp1:
            st.markdown(f'<p class="sb-sub" style="margin-bottom:8px;">High-Risk Routes (Longest Lead Times)</p>', unsafe_allow_html=True)
            hr = (dff.groupby(['Region','Ship Mode','Factory'])['Lead Time']
                  .mean().reset_index().sort_values('Lead Time',ascending=False).head(8).round(1))
            st.dataframe(hr, use_container_width=True, hide_index=True)
        with rp2:
            st.markdown(f'<p class="sb-sub" style="margin-bottom:8px;">Profit Alerts (Lowest Margin Products)</p>', unsafe_allow_html=True)
            lm = dff.copy(); lm['Margin %'] = (lm['Gross Profit']/lm['Sales']*100).round(1)
            al = lm.groupby(['Product Name','Factory'])['Margin %'].mean().reset_index().sort_values('Margin %').head(8)
            st.dataframe(al, use_container_width=True, hide_index=True)
