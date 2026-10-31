import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="COVID-19 Analytics Platform | Apache Spark Cluster",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "processed" / "covid_data_parquet"
OUTPUT_DIR = DATA_DIR / "output"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

st.markdown("""
<style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a2e40;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #5a6b7c;
        margin-bottom: 1.2rem;
    }
    .kpi-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: left;
    }
    .cluster-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #edf2f7;
        color: #2d3748;
        border: 1px solid #cbd5e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px;
        border-radius: 4px;
        background-color: #f1f5f9;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset():
    if not PARQUET_PATH.exists():
        return None
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def get_country_snapshot(df):
    summary = df.groupby(["continent", "location", "iso_code"]).agg({
        "total_cases": "max",
        "total_deaths": "max",
        "total_cases_per_million": "max",
        "total_deaths_per_million": "max",
        "people_fully_vaccinated_per_hundred": "max",
        "population": "max",
        "median_age": "first",
        "gdp_per_capita": "first",
        "population_density": "first",
        "hospital_beds_per_thousand": "first",
        "diabetes_prevalence": "first",
        "cardiovasc_death_rate": "first",
        "human_development_index": "first"
    }).reset_index()

    summary["case_fatality_ratio_pct"] = (
        (summary["total_deaths"] / summary["total_cases"]) * 100
    ).round(2)
    return summary


df = load_dataset()

with st.sidebar:
    st.markdown("### System Topology")
    st.markdown('<span class="cluster-tag">1 Master Node</span> <span class="cluster-tag">2 Worker Nodes</span>', unsafe_allow_html=True)
    st.caption("Engine: Apache Spark 3.5 Standalone Cluster")
    st.divider()

    st.markdown("### Global Filters")
    continents = ["All Continents"] + sorted(list(df["continent"].dropna().unique())) if df is not None else ["All Continents"]
    selected_continent = st.selectbox("Continent:", continents)

    date_min = df["date"].min().date() if df is not None else None
    date_max = df["date"].max().date() if df is not None else None

    if date_min and date_max:
        selected_date_range = st.date_input(
            "Timeline Window:",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )
    else:
        selected_date_range = None

    st.divider()
    st.caption(f"Cluster Ingestion Records: {len(df):,}" if df is not None else "No data loaded")
    st.caption("Storage Layer: Columnar Parquet (Partitioned)")

st.markdown('<div class="main-title">COVID-19 Big Data Analytics Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Distributed analysis of epidemiological growth patterns and multi-variable correlations on a 3-node Apache Spark cluster.</div>',
    unsafe_allow_html=True
)

if df is None:
    st.error("Processed Parquet dataset not found. Execute `./scripts/run_pipeline.sh` to initialize the data pipeline.")
    st.stop()

filtered_df = df.copy()
if selected_continent != "All Continents":
    filtered_df = filtered_df[filtered_df["continent"] == selected_continent]

if selected_date_range and len(selected_date_range) == 2:
    start_d, end_d = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.to_datetime(start_d)) & 
        (filtered_df["date"] <= pd.to_datetime(end_d))
    ]

country_summary = get_country_snapshot(filtered_df)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Cases", f"{country_summary['total_cases'].sum():,.0f}")
with col2:
    st.metric("Total Deaths", f"{country_summary['total_deaths'].sum():,.0f}")
with col3:
    st.metric("Avg Fatality Rate", f"{country_summary['case_fatality_ratio_pct'].mean():.2f}%")
with col4:
    st.metric("Monitored Regions", f"{country_summary['location'].nunique()}")
with col5:
    st.metric("Avg Full Vaccination", f"{country_summary['people_fully_vaccinated_per_hundred'].dropna().mean():.1f}%")

st.markdown("---")

tabs = st.tabs([
    "Overview & Spatial Distribution",
    "Regional Growth Trajectories",
    "Case Correlation Matrix",
    "Epidemiological Latency",
    "Cluster Operations"
])


# =============================================================================
# Tab 1: Overview & Spatial Distribution
# =============================================================================
with tabs[0]:
    st.subheader("Global Spatial Distribution")
    col_map1, col_map2 = st.columns([3, 1])
    with col_map2:
        map_metric = st.selectbox(
            "Selected Metric:",
            [
                ("total_cases", "Total Confirmed Cases"),
                ("total_deaths", "Total Deaths"),
                ("total_cases_per_million", "Cases Per Million"),
                ("total_deaths_per_million", "Deaths Per Million"),
                ("people_fully_vaccinated_per_hundred", "Fully Vaccinated (%)"),
                ("case_fatality_ratio_pct", "Case Fatality Rate (%)")
            ],
            format_func=lambda x: x[1]
        )

    metric_col, metric_label = map_metric
    fig_map = px.choropleth(
        country_summary,
        locations="iso_code",
        color=metric_col,
        hover_name="location",
        hover_data={"iso_code": False, "total_cases": ":,", "total_deaths": ":,", "population": ":,"},
        color_continuous_scale="Reds" if "death" in metric_col else "Blues",
        title=f"Global Indicator: {metric_label}",
        projection="natural earth"
    )
    fig_map.update_layout(margin={"r": 0, "t": 35, "l": 0, "b": 0}, coloraxis_colorbar={"title": ""})
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Regional Rankings")
    sort_metric = st.selectbox(
        "Sort Regions by:",
        ["total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million", "people_fully_vaccinated_per_hundred"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    top_countries = country_summary.sort_values(by=sort_metric, ascending=False).head(15)

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        fig_bar = px.bar(
            top_countries,
            x=sort_metric,
            y="location",
            orientation="h",
            color=sort_metric,
            color_continuous_scale="Blues",
            title=f"Top 15 Countries by {sort_metric.replace('_', ' ').title()}",
            labels={"location": "Country", sort_metric: sort_metric.replace('_', ' ').title()}
        )
        fig_bar.update_layout(yaxis={"autorange": "reversed"}, margin={"r": 10, "t": 35, "l": 10, "b": 10})
        st.plotly_chart(fig_bar, use_container_width=True)

    with table_col:
        st.dataframe(
            top_countries[["location", "total_cases", "total_deaths", "total_cases_per_million", "people_fully_vaccinated_per_hundred"]].rename(columns={
                "location": "Country",
                "total_cases": "Cases",
                "total_deaths": "Deaths",
                "total_cases_per_million": "Cases/M",
                "people_fully_vaccinated_per_hundred": "Vax %"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# Tab 2: Regional Growth Trajectories
# =============================================================================
with tabs[1]:
    st.subheader("Regional Trajectory Comparison")
    g_col1, g_col2, g_col3 = st.columns([2, 1, 1])
    available_countries = sorted(list(filtered_df["location"].dropna().unique()))
    default_countries = [c for c in ["United States", "India", "France", "Germany", "Brazil"] if c in available_countries]

    with g_col1:
        chosen_countries = st.multiselect("Selected Countries:", options=available_countries, default=default_countries[:3])
    with g_col2:
        growth_metric = st.selectbox(
            "Metric:",
            [
                ("new_cases_smoothed", "7-Day Smoothed Cases"),
                ("new_cases", "Daily New Cases"),
                ("new_deaths_smoothed", "7-Day Smoothed Deaths"),
                ("new_deaths", "Daily New Deaths"),
                ("new_cases_smoothed_per_million", "Cases/Million (Smoothed)")
            ],
            format_func=lambda x: x[1]
        )
    with g_col3:
        scale_type = st.radio("Axis Scale:", ["Linear", "Logarithmic"], horizontal=True)

    if chosen_countries:
        growth_subset = filtered_df[filtered_df["location"].isin(chosen_countries)].sort_values(by=["location", "date"])
        fig_growth = px.line(
            growth_subset,
            x="date",
            y=growth_metric[0],
            color="location",
            title=f"Comparative Trajectory: {growth_metric[1]}",
            log_y=(scale_type == "Logarithmic"),
            labels={"date": "Timeline", growth_metric[0]: growth_metric[1], "location": "Country"}
        )
        fig_growth.update_layout(hovermode="x unified", legend={"title": ""})
        st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("Growth Factor and Phase Classification")
        focus_country = st.selectbox("Detailed Analysis for Country:", chosen_countries)
        focus_df = filtered_df[filtered_df["location"] == focus_country].sort_values(by="date").copy()
        
        focus_df["growth_factor"] = focus_df["new_cases_smoothed"] / focus_df["new_cases_smoothed"].shift(1)
        focus_df = focus_df.dropna(subset=["growth_factor"])
        focus_df["phase"] = np.where(
            focus_df["growth_factor"] > 1.05, "Expansion (Gt > 1.05)",
            np.where(focus_df["growth_factor"] < 0.95, "Decline (Gt < 0.95)", "Equilibrium (0.95-1.05)")
        )

        gf_chart_col, gf_pie_col = st.columns([3, 1])
        with gf_chart_col:
            fig_gf = px.scatter(
                focus_df,
                x="date",
                y="growth_factor",
                color="phase",
                color_discrete_map={
                    "Expansion (Gt > 1.05)": "#c0392b",
                    "Decline (Gt < 0.95)": "#27ae60",
                    "Equilibrium (0.95-1.05)": "#2980b9"
                },
                title=f"Daily Growth Factor (Gt) Dynamics: {focus_country}",
                labels={"growth_factor": "Growth Factor (Gt)", "date": "Date"}
            )
            fig_gf.add_hline(y=1.0, line_dash="dash", line_color="#333333")
            fig_gf.update_yaxes(range=[0.5, 2.0])
            st.plotly_chart(fig_gf, use_container_width=True)

        with gf_pie_col:
            phase_counts = focus_df["phase"].value_counts().reset_index()
            phase_counts.columns = ["Phase", "Days"]
            fig_pie = px.pie(
                phase_counts,
                values="Days",
                names="Phase",
                title="Phase Distribution",
                hole=0.45,
                color="Phase",
                color_discrete_map={
                    "Expansion (Gt > 1.05)": "#c0392b",
                    "Decline (Gt < 0.95)": "#27ae60",
                    "Equilibrium (0.95-1.05)": "#2980b9"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Peak Infection Wave Detection")
    peaks_file = OUTPUT_DIR / "country_infection_peaks.csv"
    if peaks_file.exists():
        peaks_data = pd.read_csv(peaks_file)
        st.dataframe(
            peaks_data.head(20).rename(columns={
                "location": "Country",
                "peak_date": "Peak Date",
                "peak_daily_cases": "Max Single-Day Cases",
                "peak_rolling_cases": "Max 7-Day Rolling",
                "total_cases": "Total Cases",
                "cases_per_100k_pop": "Cases / 100k"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# Tab 3: Case Correlation Matrix
# =============================================================================
with tabs[2]:
    st.subheader("Correlation Analysis")
    st.caption("Pairwise statistical correlations across demographic, healthcare, and transmission indicators.")

    correlation_vars = [
        "total_cases_per_million",
        "total_deaths_per_million",
        "people_fully_vaccinated_per_hundred",
        "case_fatality_ratio_pct",
        "median_age",
        "gdp_per_capita",
        "population_density",
        "hospital_beds_per_thousand",
        "diabetes_prevalence",
        "cardiovasc_death_rate",
        "human_development_index"
    ]

    c_opt1, c_opt2 = st.columns([3, 1])
    with c_opt1:
        selected_vars = st.multiselect(
            "Selected Variables:",
            options=correlation_vars,
            default=[
                "total_cases_per_million",
                "total_deaths_per_million",
                "people_fully_vaccinated_per_hundred",
                "case_fatality_ratio_pct",
                "median_age",
                "gdp_per_capita",
                "hospital_beds_per_thousand",
                "human_development_index"
            ],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with c_opt2:
        corr_method = st.radio("Method:", ["pearson", "spearman"], horizontal=True)

    if len(selected_vars) >= 2:
        corr_data = country_summary[selected_vars].dropna()
        computed_corr = corr_data.corr(method=corr_method).round(3)

        fig_heat = px.imshow(
            computed_corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Pairwise {corr_method.capitalize()} Correlation Heatmap (n={len(corr_data)})",
            labels=dict(color="r")
        )
        fig_heat.update_xaxes(side="bottom")
        fig_heat.update_layout(height=520)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Bivariate Regression")
    b_col1, b_col2, b_col3 = st.columns([1, 1, 1])
    with b_col1:
        x_axis = st.selectbox("X-Axis:", correlation_vars, index=2, format_func=lambda x: x.replace("_", " ").title())
    with b_col2:
        y_axis = st.selectbox("Y-Axis:", correlation_vars, index=3, format_func=lambda x: x.replace("_", " ").title())
    with b_col3:
        color_by = st.selectbox("Categorize Points By:", ["continent", "None"])

    scatter_df = country_summary.dropna(subset=[x_axis, y_axis]).copy()
    if "fatality" in y_axis:
        scatter_df = scatter_df[scatter_df[y_axis] < 10]

    r_val = scatter_df[x_axis].corr(scatter_df[y_axis])

    try:
        import statsmodels.api as sm
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False

    fig_scatter = px.scatter(
        scatter_df,
        x=x_axis,
        y=y_axis,
        color=color_by if color_by != "None" else None,
        hover_name="location",
        size="population" if "population" in scatter_df else None,
        trendline="ols" if has_statsmodels else None,
        title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()} (Pearson r = {r_val:.3f})",
        labels={x_axis: x_axis.replace('_', ' ').title(), y_axis: y_axis.replace('_', ' ').title()}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# Tab 4: Epidemiological Latency
# =============================================================================
with tabs[3]:
    st.subheader("Cross-Correlation Time-Lag Analysis")
    st.caption("Quantifies the temporal latency between infection surges and subsequent mortality outcomes.")

    l_col1, l_col2 = st.columns([2, 1])
    with l_col1:
        lag_scope = st.selectbox("Geographic Scope:", ["Global Aggregation"] + sorted(list(df["location"].dropna().unique())))
    with l_col2:
        max_lag_days = st.slider("Time-Lag Range (Days):", min_value=14, max_value=42, value=35, step=7)

    if lag_scope == "Global Aggregation":
        lag_input = df.groupby("date").agg({"new_cases": "sum", "new_deaths": "sum"}).reset_index()
    else:
        lag_input = df[df["location"] == lag_scope][["date", "new_cases", "new_deaths"]].sort_values("date")

    lags_range = list(range(0, max_lag_days + 1, 1))
    lag_corrs = []

    for lag in lags_range:
        shifted_deaths = lag_input["new_deaths"].shift(-lag)
        valid = lag_input["new_cases"].notnull() & shifted_deaths.notnull()
        if valid.sum() > 30:
            r = np.corrcoef(lag_input.loc[valid, "new_cases"], shifted_deaths[valid])[0, 1]
            lag_corrs.append({"lag_days": lag, "correlation_r": r})
        else:
            lag_corrs.append({"lag_days": lag, "correlation_r": np.nan})

    lag_res_df = pd.DataFrame(lag_corrs).dropna()

    if not lag_res_df.empty:
        best_row = lag_res_df.loc[lag_res_df["correlation_r"].idxmax()]
        best_lag = int(best_row["lag_days"])
        best_r = float(best_row["correlation_r"])

        col_l_chart, col_l_metric = st.columns([3, 1])
        with col_l_chart:
            fig_lag = px.line(
                lag_res_df,
                x="lag_days",
                y="correlation_r",
                markers=True,
                title=f"Latency Profile: Cases(t) vs Deaths(t + lag) - {lag_scope}",
                labels={"lag_days": "Shift (Days)", "correlation_r": "Pearson Correlation (r)"}
            )
            fig_lag.add_vline(x=best_lag, line_dash="dash", line_color="#c0392b")
            st.plotly_chart(fig_lag, use_container_width=True)

        with col_l_metric:
            st.markdown(f"""
            <div class="kpi-container">
                <small style="color: #64748b; font-weight: 600;">PEAK CORRELATION</small>
                <h2 style="color: #1e3a5f; margin: 4px 0;">r = {best_r:.3f}</h2>
                <p style="margin: 0;"><b>Optimal Latency Shift:</b> {best_lag} Days</p>
                <hr style="margin: 10px 0; border: none; border-top: 1px solid #e2e8f0;">
                <small style="color: #475569;">Daily infection spikes statistically anticipate mortality surges by {best_lag} days.</small>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Tab 5: Cluster Operations
# =============================================================================
with tabs[4]:
    st.subheader("Cluster Topology and Diagnostics")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### HDFS / Parquet Storage Partitions")
        if PARQUET_PATH.exists():
            partitions = sorted([p.name for p in PARQUET_PATH.iterdir() if p.is_dir()])
            st.success(f"Storage Layer: Active ({len(partitions)} Partitions)")
            st.write("Continent Partitions:", partitions)
        else:
            st.warning("Parquet dataset not found.")

    with col_d2:
        st.markdown("#### Distributed Nodes Configuration")
        st.code("""
Master Node: spark-master (Port 7077, Web UI 8080)
Worker Node 1: spark-worker-1 (2 Cores, 2 GB RAM)
Worker Node 2: spark-worker-2 (2 Cores, 2 GB RAM)
Storage Format: Columnar Partitioned Parquet
Cluster Mode: Apache Spark Standalone
        """, language="yaml")

    st.divider()
    st.markdown("#### Pipeline Execution")
    st.caption("Submits data cleaning, regional growth, and correlation analysis across the 3-node Spark cluster.")

    if st.button("Run Cluster Pipeline", type="primary"):
        run_script = SCRIPTS_DIR / "run_pipeline.sh"
        with st.spinner("Dispatching tasks across Master and Worker nodes..."):
            result = subprocess.run([str(run_script)], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Distributed pipeline completed successfully.")
                st.info("View completed stages in the Spark Master Web UI at http://localhost:8080")
                st.text_area("Console Output:", result.stdout, height=180)
                st.cache_data.clear()
            else:
                st.error("Execution error encountered:")
                st.text_area("Error Details:", result.stderr, height=180)

    st.markdown("#### Dataset Extracts")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        if (OUTPUT_DIR / "country_infection_peaks.csv").exists():
            with open(OUTPUT_DIR / "country_infection_peaks.csv", "rb") as f:
                st.download_button("Download Infection Peaks CSV", f, file_name="country_infection_peaks.csv", mime="text/csv")
    with dl2:
        if (OUTPUT_DIR / "correlation_matrix.csv").exists():
            with open(OUTPUT_DIR / "correlation_matrix.csv", "rb") as f:
                st.download_button("Download Correlation Matrix CSV", f, file_name="correlation_matrix.csv", mime="text/csv")
    with dl3:
        if (OUTPUT_DIR / "continent_growth_timeseries.csv").exists():
            with open(OUTPUT_DIR / "continent_growth_timeseries.csv", "rb") as f:
                st.download_button("Download Continent Time-Series CSV", f, file_name="continent_growth_timeseries.csv", mime="text/csv")
