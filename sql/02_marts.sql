-- Marts: aggregated tables the dashboard reads directly.
-- Percentages are recomputed from counts rather than averaging published percentages,
-- so larger departments carry proportionate weight.

CREATE OR REPLACE TABLE mart_weekly_national AS
SELECT
    week_ending,
    SUM(attendances)::BIGINT                       AS attendances,
    SUM(within_4h)::BIGINT                         AS within_4h,
    SUM(over_8h)::BIGINT                           AS over_8h,
    SUM(over_12h)::BIGINT                          AS over_12h,
    ROUND(100.0 * SUM(within_4h) / SUM(attendances), 2) AS pct_within_4h,
    ROUND(100.0 * SUM(over_12h)  / SUM(attendances), 2) AS pct_over_12h
FROM stg_weekly
GROUP BY week_ending
ORDER BY week_ending;

CREATE OR REPLACE TABLE mart_weekly_board AS
SELECT
    s.week_ending,
    s.board_code,
    COALESCE(b.board_name, s.board_code)          AS board_name,
    SUM(s.attendances)::BIGINT                     AS attendances,
    SUM(s.within_4h)::BIGINT                       AS within_4h,
    SUM(s.over_12h)::BIGINT                        AS over_12h,
    ROUND(100.0 * SUM(s.within_4h) / SUM(s.attendances), 2) AS pct_within_4h
FROM stg_weekly s
LEFT JOIN dim_board b USING (board_code)
GROUP BY ALL
ORDER BY s.week_ending, board_name;

CREATE OR REPLACE TABLE mart_weekly_hospital AS
SELECT
    s.week_ending,
    s.board_code,
    COALESCE(b.board_name, s.board_code)          AS board_name,
    s.hospital_code,
    COALESCE(h.hospital_name, s.hospital_code)    AS hospital_name,
    s.attendances,
    s.within_4h,
    s.over_12h,
    ROUND(100.0 * s.within_4h / s.attendances, 2) AS pct_within_4h
FROM stg_weekly s
LEFT JOIN dim_board b USING (board_code)
LEFT JOIN dim_hospital h USING (hospital_code);

-- Hospital league table: latest 52 weeks vs the same hospital in 2019 (pre-pandemic baseline).
CREATE OR REPLACE TABLE mart_hospital_league AS
WITH latest AS (SELECT MAX(week_ending) AS max_week FROM stg_weekly),
recent AS (
    SELECT hospital_code, SUM(attendances)::BIGINT AS attendances, SUM(within_4h)::BIGINT AS within_4h, SUM(over_12h)::BIGINT AS over_12h
    FROM stg_weekly, latest
    WHERE week_ending > max_week - INTERVAL 52 WEEK
    GROUP BY hospital_code
),
baseline AS (
    SELECT hospital_code, SUM(within_4h)::BIGINT AS within_4h, SUM(attendances)::BIGINT AS attendances
    FROM stg_weekly
    WHERE year(week_ending) = 2019
    GROUP BY hospital_code
)
SELECT
    r.hospital_code,
    COALESCE(h.hospital_name, r.hospital_code)             AS hospital_name,
    COALESCE(bd.board_name, hb.board_code)                 AS board_name,
    r.attendances                                          AS attendances_last_52w,
    ROUND(100.0 * r.within_4h / r.attendances, 1)          AS pct_within_4h_last_52w,
    ROUND(100.0 * b.within_4h / b.attendances, 1)          AS pct_within_4h_2019,
    ROUND(100.0 * r.within_4h / r.attendances
        - 100.0 * b.within_4h / b.attendances, 1)          AS change_pts,
    r.over_12h                                             AS over_12h_last_52w
FROM recent r
LEFT JOIN baseline b USING (hospital_code)
LEFT JOIN (SELECT DISTINCT hospital_code, board_code FROM stg_weekly) hb USING (hospital_code)
LEFT JOIN dim_board bd ON bd.board_code = hb.board_code
LEFT JOIN dim_hospital h USING (hospital_code)
ORDER BY pct_within_4h_last_52w DESC;
