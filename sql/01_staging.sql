-- Lookups: current health boards (the file also lists boards replaced in 2018/19) and hospitals.
CREATE OR REPLACE TABLE dim_board AS
SELECT HB AS board_code, HBName AS board_name
FROM raw_boards
WHERE HBDateArchived IS NULL;

CREATE OR REPLACE TABLE dim_hospital AS
SELECT HospitalCode AS hospital_code, HospitalName AS hospital_name
FROM raw_hospitals;

-- Staging: typed, cleaned weekly A&E activity at hospital level.
-- Uses the 'All' attendance category (Unplanned + New planned) to avoid double counting.
CREATE OR REPLACE TABLE stg_weekly AS
SELECT
    strptime(CAST(WeekEndingDate AS VARCHAR), '%Y%m%d')::DATE AS week_ending,
    HBT                                    AS board_code,
    TreatmentLocation                      AS hospital_code,
    NumberOfAttendancesEpisode             AS attendances,
    NumberWithin4HoursEpisode              AS within_4h,
    NumberOver4HoursEpisode                AS over_4h,
    NumberOver8HoursEpisode                AS over_8h,
    NumberOver12HoursEpisode               AS over_12h
FROM raw_weekly
WHERE AttendanceCategory = 'All'
  AND NumberOfAttendancesEpisode > 0;
