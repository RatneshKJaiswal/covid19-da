"""
Streamlit Web Application: COVID-19 Big Data Analytics Platform
Powered by Apache Spark & Hadoop Architectures
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="COVID-19 Big Data Analytics | Spark & Hadoop",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "processed" / "covid_data_parquet"
OUTPUT_DIR = DATA_DIR / "output"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7f9fc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #175d9e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-spark { background-color: #ffefe5; color: #e25a1c; }
    .badge-hadoop { background-color: #e5f6fd; color: #0288d1; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #175d9e !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Loads cleaned Parquet dataset into memory with caching."""
    if not PARQUET_PATH.exists():
        return None
    # Read Parquet
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_country_summary(df):
    """Computes country-level aggregate snapshots."""
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


# Load data
df = load_data()

# Sidebar Setup
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg", width=180)
    st.markdown("### **Big Data Engine Controls**")
    st.markdown(
        '<span class="badge badge-spark">Apache Spark 4.3</span> '
        '<span class="badge badge-hadoop">Hadoop Parquet</span>',
        unsafe_allow_html=True
    )
    st.divider()

    selected_continent = st.selectbox(
        "Filter by Continent:",
        ["All Continents"] + sorted(list(df["continent"].dropna().unique())) if df is not None else ["All Continents"]
    )

    date_min = df["date"].min().date() if df is not None else None
    date_max = df["date"].max().date() if df is not None else None

    if date_min and date_max:
        selected_date_range = st.date_input(
            "Observation Window:",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max
        )
    else:
        selected_date_range = None

    st.divider()
    st.markdown("#### System Information")
    st.caption("• Backend: PySpark Engine (Local Cluster)")
    st.caption("• Storage: Columnar Parquet Partitioned")
    st.caption(f"• Ingestion Records: {len(df):,}" if df is not None else "• No data loaded")

# Main Header
st.markdown('<div class="main-header">🦠 COVID-19 Big Data Analytics Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Distributed epidemiological processing, regional growth dynamics, and multi-variable case correlation analysis using <b>Apache Spark</b> & <b>Hadoop</b> storage architectures.</div>',
    unsafe_allow_html=True
)

if df is None:
    st.error("Processed Parquet data not found! Please run the ETL pipeline first via `scripts/run_pipeline.sh`.")
    st.stop()

# Filter by Continent & Date Range
filtered_df = df.copy()
if selected_continent != "All Continents":
    filtered_df = filtered_df[filtered_df["continent"] == selected_continent]

if selected_date_range and len(selected_date_range) == 2:
    start_d, end_d = selected_date_range
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.to_datetime(start_d)) & 
        (filtered_df["date"] <= pd.to_datetime(end_d))
    ]

country_summary = load_country_summary(filtered_df)

# Top KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    total_cases = country_summary["total_cases"].sum()
    st.metric("Total Cases", f"{total_cases:,.0f}")
with kpi2:
    total_deaths = country_summary["total_deaths"].sum()
    st.metric("Total Deaths", f"{total_deaths:,.0f}")
with kpi3:
    avg_cfr = country_summary["case_fatality_ratio_pct"].mean()
    st.metric("Avg Case Fatality Rate", f"{avg_cfr:.2f}%")
with kpi4:
    countries_count = country_summary["location"].nunique()
    st.metric("Countries Monitored", f"{countries_count}")
with kpi5:
    avg_vax = country_summary["people_fully_vaccinated_per_hundred"].dropna().mean()
    st.metric("Avg Vaccination Coverage", f"{avg_vax:.1f}%")

st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "🗺️ Spatial Distribution & Overview",
    "📈 Regional Growth Patterns",
    "🔬 Case Correlations Matrix",
    "⏱️ Epidemiological Latency Explorer",
    "⚡ Spark Big Data Operations"
])


# =============================================================================
# TAB 1: Global Spatial Distribution & Overview
# =============================================================================
with tabs[0]:
    st.subheader("Global & Regional Spatial Distribution")
    col_map_opt1, col_map_opt2 = st.columns([3, 1])
    with col_map_opt2:
        map_metric = st.selectbox(
            "Select Heatmap Indicator:",
            [
                ("total_cases", "Total Confirmed Cases"),
                ("total_deaths", "Total Deaths"),
                ("total_cases_per_million", "Cases Per Million Population"),
                ("total_deaths_per_million", "Deaths Per Million Population"),
                ("people_fully_vaccinated_per_hundred", "Fully Vaccinated (%)"),
                ("case_fatality_ratio_pct", "Case Fatality Ratio (CFR %)")
            ],
            format_func=lambda x: x[1]
        )

    # Plotly Choropleth Map
    metric_col, metric_label = map_metric
    fig_map = px.choropleth(
        country_summary,
        locations="iso_code",
        color=metric_col,
        hover_name="location",
        hover_data={
            "iso_code": False,
            "total_cases": ":,",
            "total_deaths": ":,",
            "people_fully_vaccinated_per_hundred": ":.1f",
            "population": ":,"
        },
        color_continuous_scale="Reds" if "death" in metric_col else "Viridis",
        title=f"Global Choropleth: {metric_label}",
        projection="natural earth"
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={"title": ""}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top Ranked Countries Leaderboard
    st.subheader("Top Affected Countries Leaderboard")
    sort_metric = st.selectbox(
        "Rank Countries by:",
        ["total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million", "people_fully_vaccinated_per_hundred"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    top_countries = country_summary.sort_values(by=sort_metric, ascending=False).head(15)
    
    col_lead_chart, col_lead_table = st.columns([3, 2])
    with col_lead_chart:
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
        fig_bar.update_layout(yaxis={"autorange": "reversed"}, margin={"r": 10, "t": 40, "l": 10, "b": 10})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_lead_table:
        st.dataframe(
            top_countries[[
                "location", "total_cases", "total_deaths", 
                "total_cases_per_million", "people_fully_vaccinated_per_hundred"
            ]].rename(columns={
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
# TAB 2: Regional Growth Patterns & Waves
# =============================================================================
with tabs[1]:
    st.subheader("Regional Growth Trajectories & Moving Averages")

    growth_col1, growth_col2, growth_col3 = st.columns([2, 1, 1])
    available_countries = sorted(list(filtered_df["location"].dropna().unique()))
    default_countries = [c for c in ["United States", "India", "France", "Germany", "Brazil"] if c in available_countries]

    with growth_col1:
        chosen_countries = st.multiselect(
            "Select Countries to Compare:",
            options=available_countries,
            default=default_countries[:3]
        )
    with growth_col2:
        growth_metric = st.selectbox(
            "Curve Metric:",
            [
                ("new_cases_smoothed", "7-Day Smoothed Cases"),
                ("new_cases", "Daily New Cases (Raw)"),
                ("new_deaths_smoothed", "7-Day Smoothed Deaths"),
                ("new_deaths", "Daily New Deaths (Raw)"),
                ("new_cases_smoothed_per_million", "Cases/Million (Smoothed)"),
            ],
            format_func=lambda x: x[1]
        )
    with growth_col3:
        scale_type = st.radio("Y-Axis Scale:", ["Linear", "Logarithmic"], horizontal=True)

    if chosen_countries:
        growth_subset = filtered_df[filtered_df["location"].isin(chosen_countries)].copy()
        growth_subset = growth_subset.sort_values(by=["location", "date"])

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

        # Single Country Epidemic Growth Factor Analysis
        st.markdown("#### Growth Factor & Epidemic Expansion Phase Analysis")
        focus_country = st.selectbox("Detailed Growth Dynamics for:", chosen_countries)
        focus_df = filtered_df[filtered_df["location"] == focus_country].sort_values(by="date").copy()
        
        # Calculate growth factor G_t = cases(t) / cases(t-1)
        focus_df["growth_factor"] = focus_df["new_cases_smoothed"] / focus_df["new_cases_smoothed"].shift(1)
        focus_df = focus_df.dropna(subset=["growth_factor"])
        focus_df["phase"] = np.where(
            focus_df["growth_factor"] > 1.05, "Exponential Growth",
            np.where(focus_df["growth_factor"] < 0.95, "Controlled / Decline", "Plateau / Equilibrium")
        )

        col_gf_chart, col_gf_dist = st.columns([3, 1])
        with col_gf_chart:
            fig_gf = px.scatter(
                focus_df,
                x="date",
                y="growth_factor",
                color="phase",
                color_discrete_map={
                    "Exponential Growth": "#e41a1c",
                    "Controlled / Decline": "#4daf4a",
                    "Plateau / Equilibrium": "#377eb8"
                },
                title=f"Daily Growth Factor ($G_t$) Trajectory for {focus_country}",
                labels={"growth_factor": "Growth Factor (G_t)", "date": "Date"}
            )
            fig_gf.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Equilibrium (G=1.0)")
            fig_gf.update_yaxes(range=[0.5, 2.0])
            st.plotly_chart(fig_gf, use_container_width=True)

        with col_gf_dist:
            phase_counts = focus_df["phase"].value_counts().reset_index()
            phase_counts.columns = ["Phase", "Days"]
            fig_pie = px.pie(
                phase_counts,
                values="Days",
                names="Phase",
                title="Epidemic Phase Distribution",
                hole=0.45,
                color="Phase",
                color_discrete_map={
                    "Exponential Growth": "#e41a1c",
                    "Controlled / Decline": "#4daf4a",
                    "Plateau / Equilibrium": "#377eb8"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Country Infection Peaks Table
    st.markdown("#### Historical Peak Infection Waves Detection")
    peaks_file = OUTPUT_DIR / "country_infection_peaks.csv"
    if peaks_file.exists():
        peaks_data = pd.read_csv(peaks_file)
        st.dataframe(
            peaks_data.head(25).rename(columns={
                "location": "Country",
                "peak_date": "Peak Surge Date",
                "peak_daily_cases": "Max Daily Cases",
                "peak_rolling_cases": "Max 7D Rolling Cases",
                "total_cases": "Cumulative Cases",
                "cases_per_100k_pop": "Cases / 100k Pop"
            }),
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 3: Multi-Variable Case Correlations Matrix
# =============================================================================
with tabs[2]:
    st.subheader("Multi-Variable Epidemiological & Socioeconomic Correlations")
    st.caption("Investigate statistical relationships between viral transmission, mortality, demographics, and healthcare infrastructure.")

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

    corr_opt1, corr_opt2 = st.columns([3, 1])
    with corr_opt1:
        selected_vars = st.multiselect(
            "Select Features for Correlation Matrix:",
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
    with corr_opt2:
        corr_method = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True)

    if len(selected_vars) >= 2:
        corr_data = country_summary[selected_vars].dropna()
        computed_corr = corr_data.corr(method=corr_method).round(3)

        fig_heat = px.imshow(
            computed_corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Pairwise {corr_method.capitalize()} Correlation Heatmap (n={len(corr_data)} countries)",
            labels=dict(color="Correlation (r)")
        )
        fig_heat.update_xaxes(side="bottom")
        fig_heat.update_layout(height=550)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Dynamic Bivariate Scatter Plot with Regression
    st.subheader("Bivariate Feature Relationship & OLS Regression")
    biv_c1, biv_c2, biv_c3 = st.columns([1, 1, 1])
    with biv_c1:
        x_axis = st.selectbox(
            "X-Axis Feature:",
            correlation_vars,
            index=2, # people_fully_vaccinated_per_hundred
            format_func=lambda x: x.replace("_", " ").title()
        )
    with biv_c2:
        y_axis = st.selectbox(
            "Y-Axis Feature:",
            correlation_vars,
            index=3, # case_fatality_ratio_pct
            format_func=lambda x: x.replace("_", " ").title()
        )
    with biv_c3:
        color_by = st.selectbox("Color Data Points By:", ["continent", "None"])

    scatter_df = country_summary.dropna(subset=[x_axis, y_axis]).copy()
    # Filter extreme outliers for clean view if CFR
    if "fatality" in y_axis:
        scatter_df = scatter_df[scatter_df[y_axis] < 10]

    # Calculate Pearson r
    r_val = scatter_df[x_axis].corr(scatter_df[y_axis])

    # Check if statsmodels is available for OLS trendline
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
        title=f"{x_axis.replace('_', ' ').title()} vs. {y_axis.replace('_', ' ').title()} (Pearson r = {r_val:.3f})",
        labels={x_axis: x_axis.replace('_', ' ').title(), y_axis: y_axis.replace('_', ' ').title()}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# TAB 4: Time-Lagged Cross-Correlation Explorer
# =============================================================================
with tabs[3]:
    st.subheader("Epidemiological Latency & Time-Lag Cross-Correlation")
    st.markdown("""
    > [!NOTE]
    > Infections precede deaths by a biological and healthcare incubation window (~1 to 3 weeks).
    > By shifting the death timeseries backwards: $\\text{Deaths}(t + \\tau)$, we reveal the exact latency where statistical correlation $\\rho(\\tau)$ reaches its global maximum.
    """)

    lag_c1, lag_c2 = st.columns([2, 1])
    with lag_c1:
        lag_scope = st.selectbox(
            "Select Country for Latency Analysis:",
            ["Global Aggregation"] + sorted(list(df["location"].dropna().unique())),
            index=0
        )
    with lag_c2:
        max_lag_days = st.slider("Max Time-Lag Range (Days):", min_value=14, max_value=45, value=35, step=7)

    # Compute lagged cross-correlation dynamically
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

        col_lag_plot, col_lag_info = st.columns([3, 1])
        with col_lag_plot:
            fig_lag = px.line(
                lag_res_df,
                x="lag_days",
                y="correlation_r",
                markers=True,
                title=f"Cross-Correlation Curve: Cases(t) vs Deaths(t + τ) for {lag_scope}",
                labels={"lag_days": "Time Shift τ (Days)", "correlation_r": "Pearson Correlation (r)"}
            )
            fig_lag.add_vline(x=best_lag, line_dash="dash", line_color="red", annotation_text=f"Peak Latency: {best_lag} Days")
            st.plotly_chart(fig_lag, use_container_width=True)

        with col_lag_info:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Peak Correlation</h4>
                <h2 style="color: #e25a1c;">r = {best_r:.3f}</h2>
                <p><b>Optimal Latency Shift:</b> {best_lag} Days</p>
                <hr>
                <small>This indicates daily case spikes statistically anticipate mortality surges by approximately <b>{best_lag} days</b>.</small>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 5: Big Data & Spark Operations Engine
# =============================================================================
with tabs[4]:
    st.subheader("Big Data Engine & Spark Storage Diagnostics")

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("#### HDFS / Parquet Storage Partitions")
        if PARQUET_PATH.exists():
            partitions = [p.name for p in PARQUET_PATH.iterdir() if p.is_dir()]
            st.success(f"Storage Layer: Active ({len(partitions)} Partitions)")
            st.write("Discovered Partitions:", partitions)
        else:
            st.warning("Parquet dataset not found.")

    with col_diag2:
        st.markdown("#### Apache Spark Configuration")
        st.code(f"""
Master: local[*]
Storage Format: Apache Parquet (Columnar Partitioned)
Driver Memory: 4GB
Executor Memory: 4GB
Python Executable: {sys.executable}
JDK Environment: /Library/Java/JavaVirtualMachines/jdk-27.jdk
        """, language="yaml")

    st.divider()
    st.markdown("#### Pipeline Execution Control")
    st.caption("Trigger an on-demand re-execution of the complete PySpark ingestion, cleaning, and analytics pipeline.")

    if st.button("🚀 Re-Run PySpark Big Data Pipeline", type="primary"):
        with st.spinner("Executing PySpark pipeline jobs (ETL, Regional Growth, Correlations)..."):
            run_script = SCRIPTS_DIR / "run_pipeline.sh"
            result = subprocess.run([str(run_script)], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Spark Big Data Pipeline executed successfully!")
                st.text_area("Pipeline Console Output:", result.stdout, height=200)
                st.cache_data.clear()
            else:
                st.error("Pipeline encountered an error:")
                st.text_area("Error Details:", result.stderr, height=200)

    st.markdown("#### Export Processed Datasets")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        if (OUTPUT_DIR / "country_infection_peaks.csv").exists():
            with open(OUTPUT_DIR / "country_infection_peaks.csv", "rb") as f:
                st.download_button(
                    "📥 Download Infection Peaks CSV",
                    f,
                    file_name="country_infection_peaks.csv",
                    mime="text/csv"
                )
    with col_dl2:
        if (OUTPUT_DIR / "correlation_matrix.csv").exists():
            with open(OUTPUT_DIR / "correlation_matrix.csv", "rb") as f:
                st.download_button(
                    "📥 Download Correlation Matrix CSV",
                    f,
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )
    with col_dl3:
        if (OUTPUT_DIR / "continent_growth_timeseries.csv").exists():
            with open(OUTPUT_DIR / "continent_growth_timeseries.csv", "rb") as f:
                st.download_button(
                    "📥 Download Continent Time-Series CSV",
                    f,
                    file_name="continent_growth_timeseries.csv",
                    mime="text/csv"
                )

