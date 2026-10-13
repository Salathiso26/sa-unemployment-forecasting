DROP VIEW IF EXISTS v_national_wide;
DROP TABLE IF EXISTS national_labour;
CREATE TABLE national_labour AS
SELECT date::DATE AS date, year::INTEGER AS year, quarter::INTEGER AS quarter,
       period, sex, metric, value::NUMERIC AS value
FROM staging_national_labour;

DROP TABLE IF EXISTS provincial_labour;
CREATE TABLE provincial_labour AS
SELECT date::DATE AS date, year::INTEGER AS year, quarter::INTEGER AS quarter,
       period, province, geo_level, metric, value::NUMERIC AS value
FROM staging_provincial_labour;

DROP TABLE IF EXISTS industry_employment;
CREATE TABLE industry_employment AS
SELECT date::DATE AS date, year::INTEGER AS year, quarter::INTEGER AS quarter,
       period, sex, industry, value::NUMERIC AS value
FROM staging_industry_employment;

DROP TABLE IF EXISTS unemployed_characteristics;
CREATE TABLE unemployed_characteristics AS
SELECT date::DATE AS date, year::INTEGER AS year, quarter::INTEGER AS quarter,
       period, metric, value::NUMERIC AS value
FROM staging_unemployed_characteristics;

DROP TABLE IF EXISTS eap_population_group;
CREATE TABLE eap_population_group AS
WITH numbered AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY ctid) AS rn
    FROM staging_eap_population_group
),
typed AS (
    SELECT (ARRAY['South Africa', 'Western Cape', 'Eastern Cape', 'Northern Cape',
                  'Free State', 'KwaZulu-Natal', 'North West', 'Gauteng',
                  'Mpumalanga', 'Limpopo'])[(rn - 1) / 8 + 1]   AS region,
           population_group, sex,
           total::NUMERIC                    AS total,
           not_economically_active::NUMERIC  AS not_economically_active,
           economically_active::NUMERIC      AS economically_active,
           employed::NUMERIC                 AS employed,
           COALESCE(unemployed::NUMERIC,
                    economically_active::NUMERIC - employed::NUMERIC) AS unemployed
    FROM numbered
)
SELECT region, population_group, sex, total, not_economically_active,
       economically_active, employed, unemployed,
       100 * unemployed / NULLIF(economically_active, 0) AS unemployment_rate
FROM typed;

CREATE INDEX idx_nat_period  ON national_labour(period, sex, metric);
CREATE INDEX idx_prov_period ON provincial_labour(period, province, metric);

SELECT 'national_labour' t, COUNT(*) n FROM national_labour
UNION ALL SELECT 'provincial_labour', COUNT(*) FROM provincial_labour
UNION ALL SELECT 'industry_employment', COUNT(*) FROM industry_employment
UNION ALL SELECT 'unemployed_characteristics', COUNT(*) FROM unemployed_characteristics
UNION ALL SELECT 'eap_population_group', COUNT(*) FROM eap_population_group;

-- #####################################################################
-- PART D: PRIMARY KEYS
-- If one of these fails with a duplicate-key error, that table has
-- repeated rows. Paste the error and the rows can be tracked down.
-- #####################################################################

ALTER TABLE national_labour             ADD PRIMARY KEY (date, sex, metric);
ALTER TABLE provincial_labour           ADD PRIMARY KEY (date, province, metric);
ALTER TABLE industry_employment         ADD PRIMARY KEY (date, sex, industry);
ALTER TABLE unemployed_characteristics  ADD PRIMARY KEY (date, metric);
ALTER TABLE eap_population_group        ADD PRIMARY KEY (region, population_group, sex);


-- #####################################################################
-- PART E: DATA QUALITY CHECKS
-- #####################################################################

-- E1. Quarter range and count (expect 2008-01-01 to 2025-10-01, 72 quarters).
SELECT MIN(date) AS first_q, MAX(date) AS last_q, COUNT(DISTINCT date) AS n_quarters
FROM national_labour;

-- E2. Inspect the EAP table. If most numbers are NULL or look misplaced
--     (for example employed bigger than total), the column mapping in the
--     Python build_eap_table function needs checking.
SELECT * FROM eap_population_group;


-- #####################################################################
-- PART F: ANALYTICAL QUERIES
-- Highlight ONE query at a time and press F5.
-- #####################################################################

-- Q1. Headline national unemployment-rate time series (Both sexes, LU1).
SELECT period, date, value AS unemployment_rate
FROM national_labour
WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
ORDER BY date;


-- Q2. Latest quarter snapshot: key headline indicators (Both sexes).
SELECT metric, value
FROM national_labour
WHERE sex = 'Both sexes'
  AND period = (SELECT MAX(period) FROM national_labour)
  AND metric IN ('employed', 'unemployed', 'labour_force',
                 'unemployment_rate', 'participation_rate', 'absorption_rate')
ORDER BY metric;


-- Q3. Year-on-year change in the unemployment rate (latest 8 quarters).
WITH r AS (
    SELECT date, period, value,
           LAG(value, 4) OVER (ORDER BY date) AS value_yoy
    FROM national_labour
    WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
)
SELECT period, value AS rate, value_yoy AS rate_year_ago,
       ROUND(value - value_yoy, 1) AS yoy_change_pp
FROM r
WHERE value_yoy IS NOT NULL
ORDER BY date DESC
LIMIT 8;


-- Q4. Gender gap in the unemployment rate over time (Women minus Men).
SELECT w.period, w.date,
       w.value AS women_rate, m.value AS men_rate,
       ROUND(w.value - m.value, 1) AS gap_pp
FROM national_labour w
JOIN national_labour m
  ON w.period = m.period AND m.sex = 'Men' AND m.metric = 'unemployment_rate'
WHERE w.sex = 'Women' AND w.metric = 'unemployment_rate'
ORDER BY w.date;


-- Q5. Provincial unemployment-rate ranking for the latest quarter.
SELECT province, value AS unemployment_rate,
       RANK() OVER (ORDER BY value DESC) AS rank_highest_first
FROM provincial_labour
WHERE metric = 'unemployment_rate'
  AND geo_level = 'province'
  AND period = (SELECT MAX(period) FROM provincial_labour)
ORDER BY rank_highest_first;


-- Q6. Provinces ranked by change in unemployment rate since 2008.
WITH first_q AS (
    SELECT province, value AS rate_2008
    FROM provincial_labour
    WHERE metric = 'unemployment_rate' AND geo_level = 'province'
      AND period = (SELECT MIN(period) FROM provincial_labour)
),
last_q AS (
    SELECT province, value AS rate_latest
    FROM provincial_labour
    WHERE metric = 'unemployment_rate' AND geo_level = 'province'
      AND period = (SELECT MAX(period) FROM provincial_labour)
)
SELECT f.province, f.rate_2008, l.rate_latest,
       ROUND(l.rate_latest - f.rate_2008, 1) AS change_pp
FROM first_q f JOIN last_q l ON f.province = l.province
ORDER BY change_pp DESC;


-- Q7. Seasonal pattern: average deviation from each year's mean
--     (same method as the Python output, so the trend does not hide it).
WITH r AS (
    SELECT year, quarter, value
    FROM national_labour
    WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
),
y AS (SELECT year, AVG(value) AS yearly_mean FROM r GROUP BY year)
SELECT r.quarter,
       ROUND(AVG(r.value - y.yearly_mean), 2) AS avg_deviation_pp,
       COUNT(*) AS n_obs
FROM r JOIN y USING (year)
GROUP BY r.quarter
ORDER BY r.quarter;


-- Q8. Employment level vs absorption rate, latest 8 quarters.
SELECT e.period,
       ROUND(e.value, 0)  AS employed_thousands,
       ROUND(a.value, 1)  AS absorption_rate
FROM national_labour e
JOIN national_labour a
  ON e.period = a.period AND a.sex = 'Both sexes' AND a.metric = 'absorption_rate'
WHERE e.sex = 'Both sexes' AND e.metric = 'employed'
ORDER BY e.date DESC
LIMIT 8;


-- Q9. Industry employment shares for the latest quarter (Both sexes).
SELECT industry,
       ROUND(value, 0) AS employed_thousands,
       ROUND(100.0 * value / SUM(value) OVER (), 1) AS share_pct
FROM industry_employment
WHERE sex = 'Both sexes'
  AND period = (SELECT MAX(period) FROM industry_employment)
ORDER BY value DESC;


-- Q10. Industries with the biggest net job change since 2008 (Both sexes).
WITH f AS (
    SELECT industry, value AS jobs_2008 FROM industry_employment
    WHERE sex = 'Both sexes' AND period = (SELECT MIN(period) FROM industry_employment)
),
l AS (
    SELECT industry, value AS jobs_latest FROM industry_employment
    WHERE sex = 'Both sexes' AND period = (SELECT MAX(period) FROM industry_employment)
)
SELECT f.industry,
       ROUND(f.jobs_2008, 0)   AS jobs_2008,
       ROUND(l.jobs_latest, 0) AS jobs_latest,
       ROUND(l.jobs_latest - f.jobs_2008, 0) AS net_change_thousands,
       ROUND(100 * (l.jobs_latest - f.jobs_2008) / f.jobs_2008, 1) AS change_pct
FROM f JOIN l ON f.industry = l.industry
ORDER BY net_change_thousands DESC;


-- Q11a. Why people are unemployed (latest quarter): one split of the unemployed.
SELECT metric, ROUND(value, 0) AS thousands,
       ROUND(100 * value / SUM(value) OVER (), 1) AS share_pct
FROM unemployed_characteristics
WHERE period = (SELECT MAX(period) FROM unemployed_characteristics)
  AND metric IN ('job_losers', 'job_leavers', 'new_entrants', 're_entrants')
ORDER BY thousands DESC;


-- Q11b. How long people have been unemployed (latest quarter): a separate split.
SELECT metric, ROUND(value, 0) AS thousands,
       ROUND(100 * value / SUM(value) OVER (), 1) AS share_pct
FROM unemployed_characteristics
WHERE period = (SELECT MAX(period) FROM unemployed_characteristics)
  AND metric IN ('long_term_unemployed', 'short_term_unemployed')
ORDER BY thousands DESC;


-- Q12. Unemployment rate by population group (national block, both sexes combined).
SELECT population_group,
       ROUND(SUM(employed), 0)                    AS employed_thousands,
       ROUND(SUM(COALESCE(unemployed, 0)), 0)     AS unemployed_thousands,
       ROUND(100.0 * SUM(COALESCE(unemployed, 0))
                   / SUM(economically_active), 1) AS unemployment_rate
FROM eap_population_group
WHERE region = 'South Africa'
GROUP BY population_group
ORDER BY unemployment_rate DESC;


-- Q13. National trend with quarter-on-quarter change and 4-quarter rolling average.
SELECT period, date,
       value AS unemployment_rate,
       ROUND(value - LAG(value) OVER (ORDER BY date), 2) AS qoq_change,
       ROUND(AVG(value) OVER (ORDER BY date
             ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 2) AS rolling_4q
FROM national_labour
WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
ORDER BY date;


-- Q14. Peak and lowest quarter of the series.
(SELECT 'peak' AS type, period, value
 FROM national_labour
 WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
 ORDER BY value DESC LIMIT 1)
UNION ALL
(SELECT 'lowest', period, value
 FROM national_labour
 WHERE sex = 'Both sexes' AND metric = 'unemployment_rate'
 ORDER BY value ASC LIMIT 1);


-- Q15. Long-term unemployment share over time.
WITH u AS (
    SELECT date, period,
           MAX(value) FILTER (WHERE metric = 'long_term_unemployed')  AS long_term,
           MAX(value) FILTER (WHERE metric = 'short_term_unemployed') AS short_term
    FROM unemployed_characteristics
    GROUP BY date, period
)
SELECT period, long_term, short_term,
       ROUND(100 * long_term / NULLIF(long_term + short_term, 0), 1) AS long_term_share_pct
FROM u
ORDER BY date;


-- PART G: REUSABLE VIEW (for dashboards and Python)


CREATE OR REPLACE VIEW v_national_wide AS
SELECT date, period, year, quarter,
       MAX(value) FILTER (WHERE metric = 'unemployment_rate')  AS unemployment_rate,
       MAX(value) FILTER (WHERE metric = 'participation_rate') AS participation_rate,
       MAX(value) FILTER (WHERE metric = 'absorption_rate')    AS absorption_rate,
       MAX(value) FILTER (WHERE metric = 'employed')           AS employed,
       MAX(value) FILTER (WHERE metric = 'unemployed')         AS unemployed
FROM national_labour
WHERE sex = 'Both sexes'
GROUP BY date, period, year, quarter;

SELECT * FROM v_national_wide ORDER BY date DESC LIMIT 5;


