
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sa_unemployment_pipeline import (
    load_data, reshape_all,
    get_national_wide, get_by_sex, get_provincial, get_industry, get_national_rate,
    make_features, time_split, build_models, evaluate, recursive_forecast,
    FEATURE_COLUMNS, FORECAST_HORIZON,
)

# Page setup and style

st.set_page_config(page_title="SA Unemployment Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

INK = "#16263d"
CORAL = "#e8623d"
TEAL = "#1f8a8a"
GRID = "#e3ddd3"

st.markdown(f"""
<style>
  .stApp {{ background: #faf8f4; }}
  section[data-testid="stSidebar"] {{ background: {INK}; }}
  section[data-testid="stSidebar"] * {{ color: #e9eef5 !important; }}
  h1, h2, h3 {{ color: {INK}; }}
  .kpi {{
     background:#fff; border:1px solid {GRID}; border-radius:14px;
     padding:18px 20px;
  }}
  .kpi .label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:1px;
     color:#8a8275; margin-bottom:6px; }}
  .kpi .value {{ font-size:2.0rem; font-weight:700; color:{INK}; line-height:1; }}
  .kpi .delta {{ font-size:.85rem; margin-top:6px; }}
  .up {{ color:{CORAL}; }} .down {{ color:{TEAL}; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white", margin=dict(t=60, l=10, r=10, b=10),
    colorway=[INK, CORAL, TEAL, "#c9a227", "#6a4c93", "#946846"],
)



# Cached data loading

@st.cache_data
def load_tables():
    """Read both Excel files and reshape them once, then reuse the result."""
    sheets = load_data()
    tables = reshape_all(sheets)
    return tables


@st.cache_data
def load_views(_tables):
    """Build the DataFrames each page needs, using the pipeline's own functions."""
    national = get_national_wide(_tables)
    by_sex = get_by_sex(_tables)
    provincial = get_provincial(_tables)
    industry = get_industry(_tables)
    eap = _tables["eap_population_group"]
    return national, by_sex, provincial, industry, eap


@st.cache_data
def run_forecast(_tables):
    """Train the models and produce the forecast, same logic as the script."""
    series = get_national_rate(_tables)
    features = make_features(series)
    X_train, X_test, y_train, y_test, train, test = time_split(features)

    results = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results.append({"model": name, **evaluate(y_test, model.predict(X_test))})
        fitted[name] = model

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]

    best_model = build_models()[best_name]
    best_model.fit(features[FEATURE_COLUMNS], features["y"])
    future = recursive_forecast(best_model, series)
    future["period"] = (future["date"].dt.year.astype(str) + "Q"
                        + future["date"].dt.quarter.astype(str))

    test_predictions = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test.values,
        "predicted": fitted[best_name].predict(X_test),
    })

    return comparison, best_name, future, test_predictions, series



# Load everything once

with st.spinner("Loading and reshaping the Excel data..."):
    tables = load_tables()
    national, by_sex, provincial, industry, eap = load_views(tables)

latest = national.iloc[-1]
yago = national.iloc[-5]


def kpi(col, label, value, delta=None, inverse=True):
    """Render one KPI card. inverse=True means a rise is bad (coral)."""
    html = ""
    if delta is not None:
        unit = "pp" if "%" in value else "k"
        bad = (delta > 0) if inverse else (delta < 0)
        cls = "up" if bad else "down"
        html = f'<div class="delta {cls}">{delta:+.1f} {unit} vs year ago</div>'
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div>{html}</div>',
                 unsafe_allow_html=True)



# Sidebar
st.sidebar.markdown("## 📊 SA Unemployment")
st.sidebar.caption("Stats SA · Quarterly Labour Force Survey")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Trends & seasonality", "Forecast", "Provinces", "Demographics"],
)
st.sidebar.markdown("---")
st.sidebar.metric("Latest quarter", latest["period"])
st.sidebar.metric("Unemployment rate", f"{latest['unemployment_rate']:.1f}%",
                  f"{latest['unemployment_rate'] - yago['unemployment_rate']:+.1f} pp YoY",
                  delta_color="inverse")


# Page: Overview

if page == "Overview":
    st.title("South Africa's labour market at a glance")
    st.caption(f"Latest data: {latest['period']}")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Unemployment rate", f"{latest['unemployment_rate']:.1f}%",
        latest["unemployment_rate"] - yago["unemployment_rate"])
    kpi(c2, "Employed", f"{latest['employed']/1000:.2f}m",
        (latest["employed"] - yago["employed"]) / 1000, inverse=False)
    kpi(c3, "Unemployed", f"{latest['unemployed']/1000:.2f}m",
        (latest["unemployed"] - yago["unemployed"]) / 1000)
    kpi(c4, "Participation rate", f"{latest['participation_rate']:.1f}%",
        latest["participation_rate"] - yago["participation_rate"], inverse=False)

    st.markdown("")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=national["date"], y=national["unemployment_rate"],
                  mode="lines", name="LU1 rate", line=dict(color=INK, width=2.5)))
    fig.add_trace(go.Scatter(
        x=national["date"],
        y=national["unemployment_rate"].rolling(4, center=True).mean(),
        mode="lines", name="4-quarter rolling mean",
        line=dict(color=CORAL, width=2, dash="dash")))
    fig.update_layout(title="Official unemployment rate (LU1), 2008 to 2025",
                      yaxis_title="%", hovermode="x unified", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    peak = national.loc[national["unemployment_rate"].idxmax()]
    trough = national.loc[national["unemployment_rate"].idxmin()]
    st.info(
        f"Peak: {peak['unemployment_rate']:.1f}% in {peak['period']}  ·  "
        f"Low: {trough['unemployment_rate']:.1f}% in {trough['period']}  ·  "
        f"18-year average: {national['unemployment_rate'].mean():.1f}%"
    )



# Page: Trends & seasonality

elif page == "Trends & seasonality":
    st.title("Trends, seasonality and the gender gap")
    tab1, tab2 = st.tabs(["Seasonality", "Gender gap"])

    with tab1:
        qmap = {1: "Q1 Jan-Mar", 2: "Q2 Apr-Jun", 3: "Q3 Jul-Sep", 4: "Q4 Oct-Dec"}
        d = national.copy()
        d["q"] = d["quarter"].map(qmap)

        c1, c2 = st.columns(2)
        fig = px.box(d, x="q", y="unemployment_rate", color="q", points="all",
                     title="Distribution by calendar quarter")
        fig.update_layout(showlegend=False, yaxis_title="%", xaxis_title="",
                          **PLOTLY_LAYOUT)
        c1.plotly_chart(fig, width="stretch")

        piv = d.pivot_table(index="year", columns="quarter",
                            values="unemployment_rate")
        fig2 = px.imshow(piv, text_auto=".1f", color_continuous_scale="OrRd",
                         aspect="auto", title="Rate by year x quarter",
                         labels=dict(color="%"))
        fig2.update_layout(**PLOTLY_LAYOUT)
        c2.plotly_chart(fig2, width="stretch")

    with tab2:
        fig = px.line(by_sex, x="date", y="unemployment_rate", color="sex",
                      title="Unemployment rate by sex",
                      labels={"unemployment_rate": "%", "date": ""})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")

        w = by_sex[by_sex.sex == "Women"].set_index("period")["unemployment_rate"]
        m = by_sex[by_sex.sex == "Men"].set_index("period")["unemployment_rate"]
        gap = (w - m).dropna()
        st.metric("Current gender gap (Women minus Men)", f"{gap.iloc[-1]:+.1f} pp")


# Page: Forecast

elif page == "Forecast":
    st.title("Forecasting the unemployment rate")
    with st.spinner("Training models..."):
        comparison, best_name, future, test_predictions, series = run_forecast(tables)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", best_name)
    c1.caption("Lowest hold-out RMSE")
    c2.metric("Backtest MAE", f"{comparison.iloc[0]['MAE']:.2f} pp")
    c3.metric("Next quarter", f"{future.iloc[0]['unemployment_rate']:.1f}%",
              f"{future.iloc[0]['unemployment_rate'] - series['unemployment_rate'].iloc[-1]:+.1f} pp",
              delta_color="inverse")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["date"], y=series["unemployment_rate"],
                  name="Actual", line=dict(color=INK, width=2)))
    fig.add_trace(go.Scatter(x=test_predictions["date"], y=test_predictions["predicted"],
                  name=f"{best_name} backtest", mode="lines+markers",
                  line=dict(color="#c9a227", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=future["date"], y=future["unemployment_rate"],
                  name="Forecast", mode="lines+markers",
                  line=dict(color=CORAL, width=3)))
    fig.update_layout(title="Recursive multi-step forecast", yaxis_title="%",
                      hovermode="x unified", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.markdown("#### Model comparison (hold-out)")
    a.dataframe(comparison.round(3).set_index("model"), width="stretch")
    b.markdown("#### Forecast values")
    b.dataframe(future[["period", "unemployment_rate"]].round(2).set_index("period"),
               width="stretch")



# Page: Provinces

elif page == "Provinces":
    st.title("Provincial comparison")
    latest_p = provincial["period"].iloc[-1]
    snap = provincial[provincial["period"] == latest_p].sort_values("value")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = px.bar(snap, x="value", y="province", orientation="h",
                     color="value", color_continuous_scale="OrRd",
                     title=f"Unemployment rate by province - {latest_p}",
                     labels={"value": "%", "province": ""})
        fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        picks = st.multiselect(
            "Compare provinces over time",
            sorted(provincial["province"].unique()),
            default=["Western Cape", "Gauteng", "Eastern Cape", "Limpopo"])
        sub = provincial[provincial["province"].isin(picks)]
        fig2 = px.line(sub, x="date", y="value", color="province",
                       title="Trend by province", labels={"value": "%", "date": ""})
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, width="stretch")

    first_p = provincial["period"].iloc[0]
    chg = (snap.set_index("province")["value"].rename("latest").to_frame()
           .join(provincial[provincial["period"] == first_p]
                 .set_index("province")["value"].rename("y2008")))
    chg["change_pp"] = (chg["latest"] - chg["y2008"]).round(1)
    st.markdown("#### Change since 2008")
    st.dataframe(chg.sort_values("change_pp", ascending=False).round(1),
                width="stretch")



# Page: Demographic
else:
    st.title("Industry mix and demographics")

    wide = industry.pivot_table(index="date", columns="industry", values="value")
    order = wide.iloc[-1].sort_values(ascending=False).index
    wide = wide[order]
    fig = go.Figure()
    for col in wide.columns:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[col], name=col,
                                 stackgroup="one", mode="lines"))
    fig.update_layout(title="Employment by industry over time (thousands)",
                      yaxis_title="Employed (000s)", hovermode="x unified",
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        latest_ind = industry[industry["date"] == industry["date"].max()] \
            .sort_values("value", ascending=False)
        fig2 = px.pie(latest_ind, values="value", names="industry", hole=.5,
                     title="Employment share, latest quarter")
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, width="stretch")
    with c2:
        rate = (eap.groupby("population_group")
                .apply(lambda g: 100 * g["unemployed"].sum() / g["economically_active"].sum())
                .reset_index(name="rate")
                .sort_values("rate", ascending=False))
        fig3 = px.bar(rate, x="population_group", y="rate", color="rate",
                     color_continuous_scale="OrRd",
                     title="Unemployment rate by population group",
                     labels={"rate": "%", "population_group": ""})
        fig3.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig3, width="stretch")

st.sidebar.markdown("---")
st.sidebar.caption("pandas · scikit-learn · MLflow · Plotly · Streamlit")
