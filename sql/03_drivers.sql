-- Drivers: board-level measures of hospital capacity and flow, aligned with A&E performance.
-- Weeks are assigned to the month their week-ending date falls in.

CREATE OR REPLACE TABLE stg_delayed AS
SELECT
    strptime(CAST(MonthOfDelay AS VARCHAR), '%Y%m')::DATE AS month,
    HBT                                                   AS board_code,
    NumberOfDelayedBedDays                                AS delayed_bed_days,
    AverageDailyNumberOfDelayedBeds                       AS avg_daily_delayed_beds
FROM raw_delayed
WHERE AgeGroup = '18plus'
  AND ReasonForDelay = 'All Delay Reasons';

-- Board totals for all acute specialties are the rows where Location is the board code.
CREATE OR REPLACE TABLE stg_beds AS
SELECT
    Quarter                        AS quarter,
    make_date(CAST(left(Quarter, 4) AS INT), 3 * CAST(right(Quarter, 1) AS INT) - 2, 1) AS quarter_start,
    HB                             AS board_code,
    AverageAvailableStaffedBeds    AS staffed_beds,
    AverageOccupiedBeds            AS occupied_beds,
    PercentageOccupancy            AS pct_occupancy
FROM raw_beds
WHERE SpecialtyName = 'All Acute'
  AND Location = HB;

CREATE OR REPLACE TABLE mart_monthly_board_drivers AS
WITH ae AS (
    SELECT
        date_trunc('month', week_ending)::DATE AS month,
        board_code,
        SUM(attendances)::BIGINT               AS attendances,
        SUM(within_4h)::BIGINT                 AS within_4h,
        SUM(over_12h)::BIGINT                  AS over_12h
    FROM stg_weekly
    GROUP BY ALL
)
SELECT
    ae.month,
    ae.board_code,
    COALESCE(b.board_name, ae.board_code)          AS board_name,
    ae.attendances,
    ROUND(100.0 * ae.within_4h / ae.attendances, 2) AS pct_within_4h,
    ae.over_12h,
    d.delayed_bed_days,
    d.avg_daily_delayed_beds
FROM ae
JOIN stg_delayed d USING (month, board_code)
LEFT JOIN dim_board b USING (board_code)
ORDER BY ae.month, board_name;

CREATE OR REPLACE TABLE mart_quarterly_board_occupancy AS
WITH ae AS (
    SELECT
        date_trunc('quarter', week_ending)::DATE AS quarter_start,
        board_code,
        SUM(attendances)::BIGINT                 AS attendances,
        SUM(within_4h)::BIGINT                   AS within_4h
    FROM stg_weekly
    GROUP BY ALL
)
SELECT
    s.quarter,
    ae.quarter_start,
    ae.board_code,
    COALESCE(b.board_name, ae.board_code)            AS board_name,
    ROUND(100.0 * ae.within_4h / ae.attendances, 2)  AS pct_within_4h,
    ROUND(s.pct_occupancy, 2)                        AS pct_occupancy,
    ROUND(s.staffed_beds, 1)                         AS staffed_beds
FROM ae
JOIN stg_beds s USING (quarter_start, board_code)
LEFT JOIN dim_board b USING (board_code)
ORDER BY ae.quarter_start, board_name;
