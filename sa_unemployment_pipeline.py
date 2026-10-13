"""Section 0: SA Unemployment Trends and Forecasting"""
# What this script does, in order:
#   Section 1  Load the two Excel workbooks and check them
#   Section 2  Reshape the wide Stats SA tables into tidy long tables
#   Section 3  Prepare the frames the later sections need
#   Section 4  EDA and visualisation (pandas, matplotlib, seaborn, plotly)
#   Section 5  Feature engineering and forecasting (scikit-learn)
#   Section 6  Experiment tracking (MLflow)

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 8
FORECAST_HORIZON = 6
LAGS = [1, 2, 3, 4]

TRENDS_XLSX = "QLFS_Trends_2008-2025Q4.xlsx"
EAP_XLSX = "Economically_active_population_QLFS_Q4_2025.xlsx"
OUTPUT_DIR = "forecast_outputs"

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
MLFLOW_URI = "file:./mlruns"
EXPERIMENT_NAME = "sa_unemployment_forecast"

QUARTER_MAP = {"Jan-Mar": 1, "Apr-Jun": 2, "Jul-Sep": 3, "Oct-Dec": 4}
PERIOD_RE = re.compile(r"(Jan-Mar|Apr-Jun|Jul-Sep|Oct-Dec)\s+(\d{4})")

LF_METRICS = {
    "Population 15-64 years": "population_15_64",
    "Population 15-64 yrs": "population_15_64",
    "Labour Force": "labour_force",
    "Employed": "employed",
    "Formal sector": "employed_formal",
    "Informal sector": "employed_informal",
    "Household sector": "employed_household",
    "Unemployed": "unemployed",
    "Outside the Labour Force": "not_economically_active",
    "Labour force participation rate": "participation_rate",
    "Employed / population ratio (Absorption)": "absorption_rate",
    "Inactivity rate": "inactivity_rate",
    "LU1- Unemployment rate": "unemployment_rate",
    "LU2 - Combined rate of unemployment and time-related underemployment": "unemployment_rate_lu2",
    "LU3 - Combined rate of unemployment and potential labour force": "unemployment_rate_lu3",
    "LU4 - Composite measure of labour underutilization": "unemployment_rate_lu4",
}

SEX_CATEGORIES = {"Both sexes", "Women", "Men"}

SECTION_SUBHEADERS = {
    "Key rates in relation to the working-age population",
    "Labour underutilization indicators (%)",
}

INDUSTRIES = [
    "Agriculture", "Mining", "Manufacturing", "Utilities", "Construction",
    "Trade", "Transport", "Finance", "Community and social services",
    "Private households", "Other",
]

PROVINCES = {
    "South Africa", "Western Cape", "Eastern Cape", "Northern Cape",
    "Free State", "KwaZulu Natal", "North West", "Gauteng",
    "Mpumalanga", "Limpopo",
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"



# Section 1: Load the raw Excel files


def load_data():
    """Load the five sheets the pipeline needs and print their shapes."""
    print("=" * 70)
    print("SECTION 1: LOADING RAW EXCEL DATA")
    print("=" * 70)

    df1 = pd.read_excel(TRENDS_XLSX, sheet_name="Table 2", header=None)
    df2 = pd.read_excel(TRENDS_XLSX, sheet_name="Table 2.3", header=None)
    df3 = pd.read_excel(TRENDS_XLSX, sheet_name="Table3.1", header=None)
    df4 = pd.read_excel(TRENDS_XLSX, sheet_name="Table4", header=None)
    df5 = pd.read_excel(EAP_XLSX, sheet_name="EAP", header=None)

    print(df1.shape, df2.shape, df3.shape, df4.shape, df5.shape)

    print("\ndf1 national      ", df1.shape, " missing:", int(df1.isna().sum().sum()))
    print("df2 provincial    ", df2.shape, " missing:", int(df2.isna().sum().sum()))
    print("df3 industry      ", df3.shape, " missing:", int(df3.isna().sum().sum()))
    print("df4 unemployed    ", df4.shape, " missing:", int(df4.isna().sum().sum()))
    print("df5 eap           ", df5.shape, " missing:", int(df5.isna().sum().sum()))

    print("\nFirst 5 rows of df1 (national):")
    print(df1.head())
    print()

    return {
        "national": df1,
        "provincial": df2,
        "industry": df3,
        "unemployed": df4,
        "eap": df5,
    }



# Section 2: Reshape the wide Stats SA tables into tidy long tables

def clean_label(label):
    """Strip a cell to a clean string; missing becomes an empty string."""
    if pd.isna(label):
        return ""
    return str(label).strip()


def to_float(value):
    """Convert a Stats SA cell to a float. Dashes, stars and blanks become None."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text_value = value.strip()
        if text_value in ("-", "\u2013", "", "*", ".."):
            return None
        text_value = text_value.replace(" ", "").replace(",", "")
        try:
            return float(text_value)
        except ValueError:
            return None
    return float(value)


def find_period_columns(row):
    """Find which columns hold quarter labels. Returns {col: (year, q, label, date)}."""
    found = {}
    for index, value in enumerate(row):
        match = PERIOD_RE.search(clean_label(value))
        if match:
            quarter_name = match.group(1)
            year = int(match.group(2))
            quarter = QUARTER_MAP[quarter_name]
            date = f"{year}-{(quarter - 1) * 3 + 1:02d}-01"
            found[index] = (year, quarter, f"{year}Q{quarter}", date)
    return found


def find_header_row(df, max_scan=6):
    """Find the row number that holds the quarter labels."""
    for i in range(min(max_scan, len(df))):
        if find_period_columns(df.iloc[i]):
            return i
    raise ValueError("No period header row found")


def reshape_block_table(df, categories, metric_map):
    """Turn one wide Stats SA sheet into long format: one row per value."""
    header_row = find_header_row(df)
    periods = find_period_columns(df.iloc[header_row])
    period_cols = list(periods.keys())

    records = []
    current_group = None

    for i in range(header_row + 1, len(df)):
        label = clean_label(df.iloc[i, 0])
        if not label:
            continue

        # A known category row switches the active group (sex or province).
        if label in categories:
            current_group = label
            continue

        # A known sub-heading is skipped but keeps the active group.
        if label in SECTION_SUBHEADERS:
            continue

        # A row with no numbers is an unknown heading, such as a metro block.
        # Reset the group so its child rows are skipped.
        is_heading = all(to_float(df.iloc[i, c]) is None for c in period_cols)
        if is_heading:
            current_group = None
            continue

        metric = metric_map.get(label)
        if metric is None or current_group is None:
            continue

        for col, (year, quarter, period, date) in periods.items():
            value = to_float(df.iloc[i, col])
            if value is not None:
                records.append({
                    "category": current_group,
                    "metric": metric,
                    "year": year,
                    "quarter": quarter,
                    "period": period,
                    "date": date,
                    "value": value,
                })

    return pd.DataFrame(records)


def build_national_table(df):
    """National labour force figures by sex."""
    long = reshape_block_table(df, SEX_CATEGORIES, LF_METRICS)
    long = long.rename(columns={"category": "sex"})
    return long[["date", "year", "quarter", "period", "sex", "metric", "value"]]


def build_provincial_table(df):
    """Labour force figures by province."""
    long = reshape_block_table(df, PROVINCES, LF_METRICS)
    long = long.rename(columns={"category": "province"})
    long["province"] = long["province"].replace({"KwaZulu Natal": "KwaZulu-Natal"})
    long["geo_level"] = long["province"].apply(
        lambda p: "national" if p == "South Africa" else "province"
    )
    return long[["date", "year", "quarter", "period", "province",
                 "geo_level", "metric", "value"]]


def build_industry_table(df):
    """Employment by industry and sex."""
    metric_map = {name: name for name in INDUSTRIES}
    long = reshape_block_table(df, SEX_CATEGORIES, metric_map)
    long = long.rename(columns={"category": "sex", "metric": "industry"})
    return long[["date", "year", "quarter", "period", "sex", "industry", "value"]]


def build_unemployed_table(df):
    """Characteristics of the unemployed, such as job losers and new entrants."""
    characteristics = {
        "Job losers": "job_losers",
        "Job leavers": "job_leavers",
        "New entrants": "new_entrants",
        "Re-entrants": "re_entrants",
        "Long-term unemployment (1 year and more)": "long_term_unemployed",
        "Short-term unemployment (less than 1 year)": "short_term_unemployed",
    }
    df = df.copy()
    df.iloc[4, 0] = "Unemployed"  # this sheet has one implicit category
    long = reshape_block_table(df, {"Unemployed"}, characteristics)
    return long[["date", "year", "quarter", "period", "metric", "value"]]


def build_eap_table(df):
    """Latest quarter snapshot by population group and sex."""
    groups = ["Black African", "Coloured", "Indian/ Asian", "White"]
    columns = {
        "Male": {"total": 1, "nea": 2, "eap": 3, "employed": 4, "unemployed": 5},
        "Female": {"total": 6, "nea": 7, "eap": 8, "employed": 9, "unemployed": 10},
    }

    records = []
    for i in range(len(df)):
        group = clean_label(df.iloc[i, 0])
        if group in groups:
            for sex, c in columns.items():
                records.append({
                    "population_group": group.replace("Indian/ Asian", "Indian/Asian"),
                    "sex": sex,
                    "total": to_float(df.iloc[i, c["total"]]),
                    "not_economically_active": to_float(df.iloc[i, c["nea"]]),
                    "economically_active": to_float(df.iloc[i, c["eap"]]),
                    "employed": to_float(df.iloc[i, c["employed"]]),
                    "unemployed": to_float(df.iloc[i, c["unemployed"]]),
                })

    out = pd.DataFrame(records)
    out["unemployment_rate"] = 100 * out["unemployed"] / out["economically_active"]
    return out


def reshape_all(sheets):
    """Run every reshaper and print the resulting row counts."""
    print("=" * 70)
    print("SECTION 2: RESHAPING INTO TIDY TABLES")
    print("=" * 70)

    tables = {
        "national_labour": build_national_table(sheets["national"]),
        "provincial_labour": build_provincial_table(sheets["provincial"]),
        "industry_employment": build_industry_table(sheets["industry"]),
        "unemployed_characteristics": build_unemployed_table(sheets["unemployed"]),
        "eap_population_group": build_eap_table(sheets["eap"]),
    }

    for name, df in tables.items():
        print(f"{name:<28} {len(df):>6} rows")
        print("  missing values:", int(df.isna().sum().sum()),
              " duplicate rows:", int(df.duplicated().sum()))
    print()
    return tables


# Section 3: Prepare the frames the EDA and forecasting sections need

def get_national_wide(tables):
    """National metrics, one column per metric, sorted by date."""
    long = tables["national_labour"]
    long = long[long["sex"] == "Both sexes"]

    wide = long.pivot_table(
        index=["date", "period", "year", "quarter"],
        columns="metric", values="value"
    ).reset_index()
    wide["date"] = pd.to_datetime(wide["date"])
    return wide.sort_values("date").reset_index(drop=True)


def get_by_sex(tables):
    """Unemployment rate split by Women and Men."""
    df = tables["national_labour"]
    df = df[(df["metric"] == "unemployment_rate")
            & (df["sex"].isin(["Women", "Men"]))].copy()

    df = df.rename(columns={"value": "unemployment_rate"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "period", "sex", "unemployment_rate"]].sort_values("date")


def get_provincial(tables):
    """Unemployment rate for the nine provinces over time."""
    df = tables["provincial_labour"]
    df = df[(df["metric"] == "unemployment_rate")
            & (df["geo_level"] == "province")].copy()

    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "period", "year", "quarter", "province", "value"]] \
        .sort_values(["province", "date"])


def get_industry(tables):
    """Employment by industry over time."""
    df = tables["industry_employment"]
    df = df[df["sex"] == "Both sexes"].copy()

    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "period", "industry", "value"]]


def get_national_rate(tables):
    """The single series the forecasting section uses."""
    df = tables["national_labour"]
    df = df[(df["sex"] == "Both sexes")
            & (df["metric"] == "unemployment_rate")].copy()

    df = df.rename(columns={"value": "unemployment_rate"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "unemployment_rate"]].sort_values("date").reset_index(drop=True)



# Section 4: EDA and visualisation

def check_quarters(national):
    """Confirm the quarterly series has no gaps."""
    print("=" * 70)
    print("SECTION 4: EDA AND VISUALISATION")
    print("=" * 70)
    print("Quarters covered:", national["period"].iloc[0], "to",
          national["period"].iloc[-1])
    print("Number of quarters:", len(national))

    expected = pd.date_range(national["date"].min(), national["date"].max(), freq="QS")
    missing = sorted(set(expected) - set(national["date"]))
    print("Missing quarters:", missing if missing else "none")
    print()


def describe_series(national):
    """Print descriptive statistics for the headline indicators."""
    columns = ["unemployment_rate", "participation_rate", "absorption_rate",
               "employed", "unemployed"]
    print("Descriptive statistics:")
    print(national[columns].describe().round(1).to_string())
    print()


def plot_trend(national):
    """Chart 1: the national unemployment rate over time."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(national["date"], national["unemployment_rate"],
            color="#2a4d7e", lw=2.4, marker="o", ms=3, label="Quarterly LU1 rate")

    rolling = national["unemployment_rate"].rolling(4, center=True).mean()
    ax.plot(national["date"], rolling, color="#d1495b", lw=2.6, ls="--",
            label="4-quarter rolling mean")

    ax.axhline(national["unemployment_rate"].mean(), color="grey", lw=1, alpha=0.6)
    ax.set_title("South Africa: official unemployment rate (LU1), 2008 to 2025")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.savefig(f"{OUTPUT_DIR}/01_national_trend.png")
    plt.close(fig)
    print("  saved 01_national_trend.png")


def plot_seasonality(national):
    """Chart 2: seasonal pattern, boxplot plus year by quarter heatmap."""
    labels = {1: "Q1\nJan-Mar", 2: "Q2\nApr-Jun",
              3: "Q3\nJul-Sep", 4: "Q4\nOct-Dec"}
    data = national.copy()
    data["q"] = data["quarter"].map(labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=data, x="q", y="unemployment_rate", hue="q",
                palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("Distribution by calendar quarter")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Unemployment rate (%)")

    pivot = data.pivot_table(index="year", columns="quarter",
                             values="unemployment_rate")
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="rocket_r",
                cbar_kws={"label": "Rate (%)"}, ax=axes[1])
    axes[1].set_title("Unemployment rate: year by quarter")
    axes[1].set_xlabel("Quarter")
    axes[1].set_ylabel("Year")

    fig.suptitle("Seasonal patterns in the unemployment rate",
                 fontweight="bold", y=1.02)
    fig.savefig(f"{OUTPUT_DIR}/02_seasonality.png")
    plt.close(fig)

    data["year_mean"] = data.groupby("year")["unemployment_rate"].transform("mean")
    deviation = (data.assign(dev=data["unemployment_rate"] - data["year_mean"])
                     .groupby("quarter")["dev"].mean())
    print("\nAverage deviation from each year's mean (percentage points):")
    print(deviation.round(2).to_string())
    print("  saved 02_seasonality.png")


def plot_decomposition(national):
    """Chart 3: split the series into trend, seasonal and residual."""
    series = national.set_index("date")["unemployment_rate"]
    trend = series.rolling(4, center=True).mean()
    detrended = series - trend
    seasonal = detrended.groupby(national["quarter"].values).transform("mean")
    seasonal.index = series.index
    residual = series - trend - seasonal

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    parts = [series, trend, seasonal, residual]
    names = ["Observed", "Trend (4-quarter MA)", "Seasonal", "Residual"]
    colours = ["#2a4d7e", "#d1495b", "#2e8b57", "#999999"]

    for ax, part, name, colour in zip(axes, parts, names, colours):
        ax.plot(part.index, part.values, color=colour, lw=2)
        ax.set_ylabel(name)

    axes[0].set_title("Additive decomposition of the unemployment rate")
    fig.savefig(f"{OUTPUT_DIR}/03_decomposition.png")
    plt.close(fig)
    print("  saved 03_decomposition.png")


def plot_gender_gap(by_sex):
    """Chart 4: unemployment rate for women compared with men."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for sex, colour in [("Women", "#c44e9d"), ("Men", "#2a7de1")]:
        subset = by_sex[by_sex["sex"] == sex]
        ax.plot(subset["date"], subset["unemployment_rate"],
                lw=2.3, label=sex, color=colour)

    ax.set_title("Unemployment rate by sex: a persistent gender gap")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.savefig(f"{OUTPUT_DIR}/04_gender_gap.png")
    plt.close(fig)
    print("  saved 04_gender_gap.png")


def plot_provincial(provincial):
    """Chart 5: provincial ranking and all provinces over time."""
    latest = provincial["period"].iloc[-1]
    snapshot = (provincial[provincial["period"] == latest]
                .sort_values("value", ascending=True))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    sns.barplot(data=snapshot, y="province", x="value", hue="province",
                palette="flare", legend=False, ax=axes[0])
    for i, value in enumerate(snapshot["value"]):
        axes[0].text(value + 0.4, i, f"{value:.1f}", va="center", fontsize=12)
    axes[0].set_title(f"Provincial unemployment rate: {latest}")
    axes[0].set_xlabel("Rate (%)")
    axes[0].set_ylabel("")

    for name, group in provincial.groupby("province"):
        axes[1].plot(group["date"], group["value"], lw=1.3, alpha=0.8, label=name)
    axes[1].set_title("Provincial unemployment rate over time")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xlabel("")
    axes[1].legend(fontsize=8, ncol=2, frameon=False, loc="upper left")

    fig.savefig(f"{OUTPUT_DIR}/05_provincial.png")
    plt.close(fig)
    print("  saved 05_provincial.png")


def plot_industry(industry):
    """Chart 6: employment by industry as a stacked area."""
    wide = industry.pivot_table(index="date", columns="industry", values="value")
    order = wide.iloc[-1].sort_values(ascending=False).index
    wide = wide[order]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(wide.index, wide.T.values, labels=wide.columns,
                 colors=sns.color_palette("Spectral", len(wide.columns)), alpha=0.9)
    ax.set_title("Employment by industry over time (thousands)")
    ax.set_ylabel("Employed (thousands)")
    ax.set_xlabel("")
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="upper left")
    fig.savefig(f"{OUTPUT_DIR}/06_industry_mix.png")
    plt.close(fig)
    print("  saved 06_industry_mix.png")


def plot_correlations(national):
    """Chart 7: correlation heatmap of the headline indicators."""
    columns = ["unemployment_rate", "participation_rate", "absorption_rate",
               "inactivity_rate", "employed", "unemployed", "labour_force"]
    correlation = national[columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation between headline indicators")
    fig.savefig(f"{OUTPUT_DIR}/07_correlations.png")
    plt.close(fig)
    print("  saved 07_correlations.png")


def plot_interactive(national, provincial):
    """Two interactive Plotly charts saved as standalone HTML files."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national["date"], y=national["unemployment_rate"],
        mode="lines+markers", name="LU1 rate",
        line=dict(color="#2a4d7e", width=2)))
    fig.add_trace(go.Scatter(
        x=national["date"],
        y=national["unemployment_rate"].rolling(4, center=True).mean(),
        mode="lines", name="4-quarter rolling mean",
        line=dict(color="#d1495b", width=2, dash="dash")))
    fig.update_layout(title="South Africa unemployment rate (interactive)",
                      yaxis_title="Rate (%)", template="plotly_white",
                      hovermode="x unified")
    fig.write_html(f"{OUTPUT_DIR}/interactive_national_trend.html")

    fig2 = px.line(provincial, x="date", y="value", facet_col="province",
                   facet_col_wrap=3, height=720,
                   title="Provincial unemployment rate over time",
                   labels={"value": "Rate (%)", "date": ""})
    fig2.update_yaxes(matches=None)
    fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig2.update_layout(template="plotly_white", showlegend=False)
    fig2.write_html(f"{OUTPUT_DIR}/interactive_provincial.html")
    print("  saved interactive_national_trend.html and interactive_provincial.html")


def run_eda(tables):
    """Run every EDA step and return the frames the forecast section needs."""
    national = get_national_wide(tables)
    by_sex = get_by_sex(tables)
    provincial = get_provincial(tables)
    industry = get_industry(tables)

    check_quarters(national)
    describe_series(national)

    plot_trend(national)
    plot_seasonality(national)
    plot_decomposition(national)
    plot_gender_gap(by_sex)
    plot_provincial(provincial)
    plot_industry(industry)
    plot_correlations(national)
    plot_interactive(national, provincial)

    national.to_csv(f"{OUTPUT_DIR}/national_clean.csv", index=False)
    print("  saved national_clean.csv")
    print()
    return national



# Section 5: Feature engineering and forecasting

FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS]
    + ["roll_mean_4", "roll_std_4", "diff_1", "diff_yoy",
       "q_1", "q_2", "q_3", "q_4", "t"]
)


def make_features(series):
    """Turn the single rate series into a table of predictors and a target y."""
    df = series.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    y = df["unemployment_rate"]

    features = pd.DataFrame({"date": df["date"], "y": y})

    # Past values of the series.
    for lag in LAGS:
        features[f"lag_{lag}"] = y.shift(lag)

    # Rolling statistics, shifted so the current quarter is never included.
    features["roll_mean_4"] = y.shift(1).rolling(4).mean()
    features["roll_std_4"] = y.shift(1).rolling(4).std()

    # Momentum.
    features["diff_1"] = y.shift(1) - y.shift(2)
    features["diff_yoy"] = y.shift(1) - y.shift(5)

    # Seasonality as four on/off columns.
    features["quarter"] = features["date"].dt.quarter
    for q in (1, 2, 3, 4):
        features[f"q_{q}"] = (features["quarter"] == q).astype(int)

    # A simple time trend.
    features["t"] = np.arange(len(features))

    features = features.drop(columns=["quarter"]).dropna().reset_index(drop=True)
    return features


def time_split(features):
    """Split by time, not at random. The last TEST_SIZE quarters are the test set."""
    train = features.iloc[:-TEST_SIZE]
    test = features.iloc[-TEST_SIZE:]
    X_train = train[FEATURE_COLUMNS]
    y_train = train["y"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["y"]
    return X_train, X_test, y_train, y_test, train, test


def build_models():
    """The four candidate models, untrained."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred):
    """Return MAE, RMSE, MAPE and R2 as a dictionary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def recursive_forecast(model, history):
    """Predict ahead one quarter at a time, feeding each prediction back in."""
    hist = history[["date", "unemployment_rate"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])

    predictions = []
    for _ in range(FORECAST_HORIZON):
        features = make_features(hist)
        last_row = features.iloc[[-1]][FEATURE_COLUMNS]
        next_date = hist["date"].iloc[-1] + pd.offsets.QuarterBegin(startingMonth=1)
        prediction = float(model.predict(last_row)[0])

        predictions.append({"date": next_date, "unemployment_rate": prediction})
        hist = pd.concat([hist, pd.DataFrame([predictions[-1]])], ignore_index=True)

    return pd.DataFrame(predictions)


def plot_forecast(series, test_predictions, future, best_name):
    """Chart 8: actual series, backtest and future forecast."""
    series = series.copy()
    series["date"] = pd.to_datetime(series["date"])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(series["date"], series["unemployment_rate"],
            color="#2a4d7e", lw=2, label="Actual")
    ax.plot(test_predictions["date"], test_predictions["predicted"],
            color="#e07b00", lw=2, ls="--", marker="o", ms=4,
            label=f"{best_name} (backtest)")
    ax.plot(future["date"], future["unemployment_rate"],
            color="#d1495b", lw=2.4, marker="D", ms=5, label="Forecast")
    ax.axvline(series["date"].iloc[-1], color="grey", ls=":", lw=1)

    ax.set_title(f"Unemployment rate forecast, best model: {best_name}")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/08_forecast.png", dpi=110)
    plt.close(fig)
    print("  saved 08_forecast.png")


def run_forecasting(tables):
    """Train the models, pick the best one and forecast ahead."""
    print("=" * 70)
    print("SECTION 5: FORECASTING")
    print("=" * 70)

    series = get_national_rate(tables)
    features = make_features(series)
    X_train, X_test, y_train, y_test, train, test = time_split(features)

    print("Training rows:", len(train), " Test rows:", len(test))
    print("Features used:", len(FEATURE_COLUMNS))
    print()

    results = {}
    fitted = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(y_test, model.predict(X_test))
        results[name] = metrics
        fitted[name] = model
        print(f"{name:<18} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
              f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.3f}")

    comparison = (pd.DataFrame(results).T
                  .sort_values("RMSE")
                  .rename_axis("model")
                  .reset_index())
    best_name = comparison.iloc[0]["model"]
    print("\nBest model by RMSE:", best_name)

    # Refit the winner on every row before forecasting the future.
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

    comparison.round(4).to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    future.to_csv(f"{OUTPUT_DIR}/future_forecast.csv", index=False)
    test_predictions.to_csv(f"{OUTPUT_DIR}/backtest_predictions.csv", index=False)

    plot_forecast(series, test_predictions, future, best_name)

    print("\nForecast for the next", FORECAST_HORIZON, "quarters:")
    print(future[["period", "unemployment_rate"]]
          .assign(unemployment_rate=lambda d: d["unemployment_rate"].round(2))
          .to_string(index=False))
    print()
    return features, train, test



# Section 6: Experiment tracking with MLflow

def run_mlflow(features, train, test):
    """Log every model, plus a naive baseline, to MLflow."""
    print("=" * 70)
    print("SECTION 6: MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["y"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["y"]

    leaderboard = []

    # Baseline: this quarter equals the same quarter last year.
    with mlflow.start_run(run_name="naive_seasonal_baseline"):
        metrics = evaluate(test["y"], test["lag_4"])
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_param("rule", "y_t = y_t-4")
        mlflow.log_metrics(metrics)
        leaderboard.append({"model": "naive_seasonal", **metrics})
        print(f"{'naive_seasonal':<18} MAE={metrics['MAE']:.3f}  "
              f"RMSE={metrics['RMSE']:.3f}  MAPE={metrics['MAPE']:.2f}%")

    for name, model in build_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(y_test, model.predict(X_test))

            mlflow.set_tag("model_type", "sklearn_regressor")
            mlflow.log_param("model_class", model.__class__.__name__)
            mlflow.log_param("n_features", len(FEATURE_COLUMNS))
            mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("test_size", len(test))

            for key, value in model.get_params().items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    mlflow.log_param(f"hp_{key}", value)

            mlflow.log_metrics(metrics)

            # Tree-based models store a sklearn.tree._tree.Tree object. Newer
            # MLflow refuses to save that unless it is listed as trusted. We
            # built these models in this script, so they are safe to save.
            mlflow.sklearn.log_model(
                model, name="model",
                input_example=X_train.iloc[:2],
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            leaderboard.append({"model": name, **metrics})
            print(f"{name:<18} MAE={metrics['MAE']:.3f}  "
                  f"RMSE={metrics['RMSE']:.3f}  MAPE={metrics['MAPE']:.2f}%  "
                  f"R2={metrics['R2']:.3f}")

    board = pd.DataFrame(leaderboard).sort_values("RMSE").reset_index(drop=True)
    board.round(4).to_csv(f"{OUTPUT_DIR}/mlflow_leaderboard.csv", index=False)

    with mlflow.start_run(run_name="experiment_summary"):
        mlflow.set_tag("best_model", board.iloc[0]["model"])
        mlflow.log_metric("best_rmse", float(board.iloc[0]["RMSE"]))
        mlflow.log_artifact(f"{OUTPUT_DIR}/mlflow_leaderboard.csv")

    print("\nLeaderboard sorted by RMSE:")
    print(board.round(4).to_string(index=False))
    print("\nBest overall:", board.iloc[0]["model"])
    print("View the runs with:")
    print("  mlflow ui --backend-store-uri ./mlruns")
    print()


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sheets = load_data()
    tables = reshape_all(sheets)

    run_eda(tables)
    features, train, test = run_forecasting(tables)
    run_mlflow(features, train, test)

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Charts and CSVs are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
