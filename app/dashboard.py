"""Scotland A&E waiting times dashboard.

Run with:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis  # noqa: E402
from src.pipeline import DB_PATH, main as build_warehouse  # noqa: E402

TARGET = 95.0
# Colour-blind-safe series colours, plus grey for reference lines.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#8a8984"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Scotland A&E Waiting Times", layout="wide")


if not DB_PATH.exists():
    with st.spinner("Building data warehouse and fitting forecast models (about a minute)..."):
        build_warehouse()


@st.cache_data
def load(table: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(f"SELECT * FROM {table}").df()


def style(fig: go.Figure, y_title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    return fig


national = load("mart_weekly_national")
board = load("mart_weekly_board")
league = load("mart_hospital_league")

latest_week = national["week_ending"].max()
last_52 = national[national["week_ending"] > latest_week - pd.Timedelta(weeks=52)]
prev_52 = national[
    (national["week_ending"] <= latest_week - pd.Timedelta(weeks=52))
    & (national["week_ending"] > latest_week - pd.Timedelta(weeks=104))
]


def pct(df: pd.DataFrame) -> float:
    return 100 * df["within_4h"].sum() / df["attendances"].sum()


st.title("Scotland A&E Waiting Times")
st.caption(
    f"Type 1 Emergency Departments, weekly data to {latest_week:%d %B %Y}. "
    "Source: Public Health Scotland open data."
)

# --- Headline KPIs -----------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Seen within 4h (52 wks)",
    f"{pct(last_52):.1f}%",
    f"{pct(last_52) - pct(prev_52):+.1f} pts vs previous year",
)
k2.metric("Gap to 95% target", f"{TARGET - pct(last_52):.1f} pts")
k3.metric(
    "12h+ waits (52 wks)",
    f"{last_52['over_12h'].sum():,}",
    f"{last_52['over_12h'].sum() - prev_52['over_12h'].sum():+,} vs previous year",
    delta_color="inverse",
)
k4.metric(
    "Attendances (52 wks)",
    f"{last_52['attendances'].sum() / 1e6:.2f}m",
    f"{100 * (last_52['attendances'].sum() / prev_52['attendances'].sum() - 1):+.1f}% vs previous year",
    delta_color="off",
)

trend_tab, season_tab, hospital_tab, drivers_tab, forecast_tab, data_tab = st.tabs(
    ["National trend", "Seasonality", "Hospitals", "What's driving it?", "Forecast", "Data"]
)

# --- National trend ------------------------------------------------------------
with trend_tab:
    boards = sorted(board["board_name"].unique())
    chosen = st.multiselect("Compare health boards (up to 3)", boards, max_selections=3)

    fig = go.Figure()
    rolling = national.set_index("week_ending")["pct_within_4h"].rolling(13, center=True).mean()
    if not chosen:
        fig.add_scatter(
            x=national["week_ending"], y=national["pct_within_4h"], name="Weekly",
            line=dict(color=SERIES[0], width=1), opacity=0.35,
            hovertemplate="%{y:.1f}%",
        )
        fig.add_scatter(
            x=rolling.index, y=rolling.values, name="Scotland (13-week average)",
            line=dict(color=SERIES[0], width=2), hovertemplate="%{y:.1f}%",
        )
    else:
        fig.add_scatter(
            x=rolling.index, y=rolling.values, name="Scotland (13-week average)",
            line=dict(color=MUTED, width=2), hovertemplate="%{y:.1f}%",
        )
        # Colour follows the board's position in the selection, which the user controls.
        for colour, name in zip(SERIES, chosen):
            b = board[board["board_name"] == name].set_index("week_ending")["pct_within_4h"]
            b = b.rolling(13, center=True).mean()
            fig.add_scatter(
                x=b.index, y=b.values, name=f"{name} (13-week average)",
                line=dict(color=colour, width=2), hovertemplate="%{y:.1f}%",
            )
    fig.add_hline(
        y=TARGET, line=dict(color=MUTED, width=1, dash="dash"),
        annotation_text="95% target", annotation_position="top left",
    )
    st.plotly_chart(style(fig, "% seen within 4 hours"), width="stretch")

    fig = go.Figure()
    fig.add_bar(
        x=national["week_ending"], y=national["over_12h"], name="Waits over 12 hours",
        marker=dict(color=SERIES[0], line_width=0), hovertemplate="%{y:,}",
    )
    st.subheader("Patients waiting over 12 hours, per week")
    st.plotly_chart(style(fig, "Patients", height=320), width="stretch")

# --- Seasonality -----------------------------------------------------------------
with season_tab:
    st.markdown(
        "Share of patients seen within 4 hours, by month. Winter pressure shows as darker "
        "(worse) cells in December to March, but since 2021 the summer months are no better "
        "than pre-pandemic winters."
    )
    monthly = national.assign(
        year=national["week_ending"].dt.year, month=national["week_ending"].dt.month
    ).groupby(["year", "month"])[["within_4h", "attendances"]].sum()
    monthly["pct"] = 100 * monthly["within_4h"] / monthly["attendances"]
    grid = monthly["pct"].unstack("month")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Reversed scale: darker = fewer patients seen in time.
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=months[: grid.shape[1]], y=grid.index.astype(str),
        colorscale=SEQ_BLUE[::-1], xgap=2, ygap=2,
        text=grid.map(lambda v: "" if pd.isna(v) else f"{v:.1f}").values, texttemplate="%{text}",
        hovertemplate="%{x} %{y}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% in 4h"),
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    st.plotly_chart(style(fig, "", height=520), width="stretch")

# --- Hospitals -----------------------------------------------------------------
with hospital_tab:
    st.markdown(
        "Each hospital's 4-hour performance over the latest 52 weeks compared with 2019. "
        "Every hospital is worse than before the pandemic; the gap shows by how much."
    )
    lg = league.dropna(subset=["pct_within_4h_2019"]).sort_values("pct_within_4h_last_52w")
    label = lg["hospital_name"] + " (" + lg["board_name"] + ")"
    fig = go.Figure()
    for _, row in lg.iterrows():
        fig.add_shape(
            type="line", x0=row["pct_within_4h_2019"], x1=row["pct_within_4h_last_52w"],
            y0=row["hospital_name"] + " (" + row["board_name"] + ")",
            y1=row["hospital_name"] + " (" + row["board_name"] + ")",
            line=dict(color=MUTED, width=2),
        )
    fig.add_scatter(
        x=lg["pct_within_4h_2019"], y=label, mode="markers", name="2019",
        marker=dict(color=MUTED, size=9), hovertemplate="2019: %{x:.1f}%<extra></extra>",
    )
    fig.add_scatter(
        x=lg["pct_within_4h_last_52w"], y=label, mode="markers", name="Last 52 weeks",
        marker=dict(color=SERIES[0], size=10), hovertemplate="Last 52 weeks: %{x:.1f}%<extra></extra>",
    )
    fig.add_vline(x=TARGET, line=dict(color=MUTED, width=1, dash="dash"))
    fig = style(fig, "", height=26 * len(lg) + 80)
    fig.update_layout(hovermode="closest", xaxis_title="% seen within 4 hours")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        league.rename(columns={
            "hospital_name": "Hospital", "board_name": "Health board",
            "attendances_last_52w": "Attendances (52w)",
            "pct_within_4h_last_52w": "% in 4h (52w)", "pct_within_4h_2019": "% in 4h (2019)",
            "change_pts": "Change (pts)", "over_12h_last_52w": "12h+ waits (52w)",
        }).drop(columns="hospital_code"),
        hide_index=True, width="stretch",
    )

# --- Drivers -----------------------------------------------------------------------
@st.cache_data
def driver_results():
    return analysis.board_change(), analysis.panel_models(), analysis.occupancy_model()


brk = analysis.find_break(national.set_index("week_ending")["pct_within_4h"])

with drivers_tab:
    changes, models, occ = driver_results()
    rho_att = changes["attendance_change_pct"].corr(changes["change_pts"], method="spearman")
    rho_size = changes["attendances_base"].corr(changes["change_pts"], method="spearman")

    st.markdown(
        f"**The fall was a step change, not a drift.** A change-point search on the weekly series puts "
        f"the break at **{brk['date']:%B %Y}**: performance averaged {brk['mean_before']:.1f}% before "
        f"and {brk['mean_after']:.1f}% after."
    )
    st.markdown(
        f"**It is not demand.** Attendances are back at 2019 levels, and boards where attendances grew "
        f"most declined *least* (rank correlation {rho_att:+.2f})."
    )
    st.markdown(
        f"**Delayed discharges rose with the decline nationally, but do not explain the differences "
        f"between boards.** The largest boards, already closest to their limits in 2019, fell furthest "
        f"(rank correlation between board size and change: {rho_size:+.2f})."
    )

    left, right = st.columns(2)
    for col, x, title, colour in (
        (left, "attendance_change_pct", "Change in attendances since 2019 (%)", SERIES[0]),
        (right, "delayed_change_pct", "Change in delayed discharges since 2019 (%)", SERIES[1]),
    ):
        # Orkney's delays rose from 1 to 13 beds (+1,233%), which would squash every other board.
        plot = changes.drop(index="NHS Orkney") if x == "delayed_change_pct" else changes
        fig = go.Figure(go.Scatter(
            x=plot[x], y=plot["change_pts"], mode="markers+text",
            text=plot.index.str.replace("NHS ", ""), textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=(plot["attendances_base"] / 8000).clip(lower=8), color=colour,
                        line=dict(color="white", width=1)),
            hovertemplate="%{text}<br>x: %{x:.1f}%<br>Change: %{y:.1f} pts<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        fig = style(fig, "Change in % seen within 4h (pts)", height=440)
        fig.update_layout(xaxis_title=title, hovermode="closest", showlegend=False)
        col.plotly_chart(fig, width="stretch")
    st.caption("Each dot is a health board; size shows its 2019 attendances. Orkney is left out of "
               "the right-hand chart because its delays rose from 1 to 13 beds (+1,233%).")

    st.subheader("Regression results")
    st.markdown(
        "Fixed-effects regressions of monthly board performance, with standard errors clustered by board. "
        "**Model A** compares each board with itself over time. **Model B** also removes anything that "
        "hit every board in the same month, so it asks whether boards whose delays rose *more* did *worse*."
    )
    st.dataframe(
        models.rename(columns={
            "model": "Model", "variable": "Variable",
            "effect_of_10pct_rise_pts": "Effect of a 10% rise (pts)", "p_value": "p-value",
            "r_squared": "R²", "n": "Board-months",
        }).style.format({"Effect of a 10% rise (pts)": "{:+.2f}", "p-value": "{:.3f}", "R²": "{:.2f}"}),
        hide_index=True, width="stretch",
    )
    st.markdown(
        f"Bed occupancy (quarterly, from late 2020 only): raw correlation with performance "
        f"{occ['raw_correlation']:+.2f}, but {occ['effect_per_pt_occupancy']:+.2f} pts per point of "
        f"occupancy (p = {occ['p_value']:.2f}) once board and quarter effects are included."
    )
    st.info(
        "These are associations in observational data, not causal effects, and there are only 14 "
        "health boards. The full analysis is in notebooks/01_drivers.ipynb."
    )

# --- Forecast -----------------------------------------------------------------------
with forecast_tab:
    fc = load("forecast")
    bt = load("forecast_backtest")
    best = fc["model"].iloc[0]
    scores = bt.groupby("model")["error"].agg(
        mae=lambda e: e.abs().mean(), rmse=lambda e: (e ** 2).mean() ** 0.5
    ).sort_values("mae")
    naive_mae = scores.loc["Naive (last value)", "mae"]

    st.markdown(
        f"12-week forecast of the national share of patients seen within 4 hours. Models are trained "
        f"only on data since the {brk['date']:%B %Y} break, because earlier weeks come from a different "
        f"regime. The best model in backtesting, **{best.lower()}**, is used for the forecast."
    )

    history = national[national["week_ending"] > latest_week - pd.Timedelta(weeks=104)]
    fig = go.Figure()
    fig.add_scatter(
        x=pd.concat([fc["week_ending"], fc["week_ending"][::-1]]),
        y=pd.concat([fc["upper_80"], fc["lower_80"][::-1]]),
        fill="toself", fillcolor="rgba(235,104,52,0.18)", line=dict(width=0),
        name="80% interval", hoverinfo="skip",
    )
    fig.add_scatter(
        x=history["week_ending"], y=history["pct_within_4h"], name="Actual",
        line=dict(color=SERIES[0], width=2), hovertemplate="%{y:.1f}%",
    )
    fig.add_scatter(
        x=fc["week_ending"], y=fc["predicted"], name="Forecast",
        line=dict(color=SERIES[1], width=2, dash="dot"), hovertemplate="%{y:.1f}%",
    )
    fig.add_hline(y=TARGET, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text="95% target", annotation_position="top left")
    st.plotly_chart(style(fig, "% seen within 4 hours"), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast average, next 12 weeks", f"{fc['predicted'].mean():.1f}%")
    c2.metric("Backtest error (MAE)", f"{scores.loc[best, 'mae']:.2f} pts")
    c3.metric("Improvement on naive baseline", f"{100 * (1 - scores.loc[best, 'mae'] / naive_mae):.0f}%")

    st.subheader("How the models compare")
    st.markdown(
        "Rolling backtest over the last two years: every 4 weeks each model is refitted using only the "
        "data available at that point and asked to forecast the next 12 weeks. Lower error is better."
    )
    by_h = bt.groupby(["model", "horizon"])["error"].apply(lambda e: e.abs().mean()).unstack(0)
    fig = go.Figure()
    for colour, name in zip(SERIES + [MUTED], scores.index):
        fig.add_scatter(x=by_h.index, y=by_h[name], name=name, mode="lines+markers",
                        line=dict(color=colour, width=2), marker=dict(size=8),
                        hovertemplate="%{y:.2f} pts")
    fig = style(fig, "Mean absolute error (pts)", height=360)
    fig.update_layout(xaxis_title="Weeks ahead")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        scores.rename(columns={"mae": "MAE (pts)", "rmse": "RMSE (pts)"}).style.format("{:.2f}"),
        width="stretch",
    )

# --- Data ------------------------------------------------------------------------
with data_tab:
    st.markdown("Weekly national figures used in the charts above.")
    st.dataframe(national.sort_values("week_ending", ascending=False), hide_index=True, width="stretch")
    st.download_button(
        "Download CSV", national.to_csv(index=False), "scotland_ae_weekly.csv", "text/csv"
    )
