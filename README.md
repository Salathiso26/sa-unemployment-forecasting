# sa-unemployment-forecasting
End-to-end analysis of South Africa’s quarterly unemployment (Stats SA QLFS, 2008-2025): EDA, PostgreSQL analysis, forecasting with scikit-learn and MLflow tracking, and an interactive Streamlit dashboard.
> I analysed South Africa's quarterly unemployment data from Stats SA, uncovered trends and seasonal patterns through EDA, loaded the data into PostgreSQL for SQL-based analysis, trained and tracked forecasting models with MLflow, and built an interactive Streamlit app with dashboards, forecasts, and provincial comparisons

**Live app:**

## Key findings

- The unemployment rate averaged 27.8% from 2008 to 2025, climbing from roughly 22% pre-2009 to 31.4% in Q4 2025, and peaked at 35.3% in Q4 2021.
- The sharp dip in Q2 2020 is a measurement artefact, not an improvement. COVID lockdowns stopped people from actively job-searching, which shrank the measured labour force rather than reflecting real recovery.
- Mild but consistent seasonality: Q3 tends to run above each year's average, Q4 below it.
- A persistent gender gap, women have faced a higher unemployment rate than men in almost every quarter of the series.
- Wide provincial dispersion: in Q4 2025 the rate ranged from 18.1% in the Western Cape to 42.5% in the Eastern Cape.
- A naive seasonal baseline (this quarter equals the same quarter last year) beat every machine learning model tested, a reminder that a strong benchmark matters more than model complexity on a persistent series like this one.

## Tech stack

- **pandas** for loading and reshaping the raw Stats SA Excel workbooks
- **PostgreSQL** for storing the tidy data and running SQL analysis
- **matplotlib, seaborn, plotly** for exploratory data analysis and visualisation
- **scikit-learn** for forecasting models and feature engineering
- **MLflow** for experiment tracking
- **Streamlit** for the interactive dashboard

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard with an overview, forecasts and provincial comparisons |
| `sa_unemployment_pipeline.py` | EDA, forecasting models and MLflow experiment tracking (pure pandas, no database) |
| `make_csv_for_sql.py` | Generates the 5 CSVs used to load PostgreSQL |
| `SA_Unemployment_forecasting.sql` | Data load, 15 analytical queries and a view |
| `QLFS_Trends_2008-2025Q4.xlsx` | Stats SA QLFS trends workbook (raw data) |
| `Economically_active_population_QLFS_Q4_2025.xlsx` | Stats SA Q4 2025 economically active population workbook (raw data) |
| `README.md` | Project documentation |

## Data source

Statistics South Africa (Stats SA), Quarterly Labour Force Survey (QLFS), Q1 2008 to Q4 2025. The data is publicly released at [statssa.gov.za](https://www.statssa.gov.za).
