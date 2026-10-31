"""
Visualization Module for COVID-19 Analysis.
Generates charts for:
1. Continent daily infection curves (7-day rolling)
2. Top country peak infections
3. Demographic and epidemiological correlation heatmap
4. Time-lagged cross-correlation curve (cases vs. deaths latency)
5. Vaccination percentage vs Case Fatality Rate (CFR)
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Apply modern aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def plot_continent_growth():
    csv_path = OUTPUT_DIR / "continent_growth_timeseries.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found. Skipping continent growth plot.")
        return

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(14, 7))
    for continent, group in df.groupby("continent"):
        plt.plot(group["date"], group["continent_rolling_cases_7d"], label=continent, linewidth=2)

    plt.title("COVID-19 Regional Growth: 7-Day Rolling Average Daily Cases by Continent", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Mean 7-Day Rolling Cases", fontsize=11)
    plt.legend(title="Continent", frameon=True)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "continent_growth_patterns.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved continent growth chart to {out_file}")

def plot_top_peaks():
    csv_path = OUTPUT_DIR / "country_infection_peaks.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found. Skipping peak infections plot.")
        return

    df = pd.read_csv(csv_path).head(12)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(df["location"][::-1], df["peak_daily_cases"][::-1] / 1000, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.title("Top 12 Countries by Peak Single-Day Infection Count", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Peak Daily Reported Cases (in Thousands)", fontsize=11)
    plt.ylabel("Country", fontsize=11)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}k",
                 ha="left", va="center", fontsize=9, color="#333333")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "top_countries_peak_cases.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved peak infections chart to {out_file}")

def plot_correlation_heatmap():
    csv_path = OUTPUT_DIR / "correlation_matrix.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found. Skipping correlation heatmap.")
        return

    df = pd.read_csv(csv_path, index_col=0)

    # Shorten column names for clean visualization
    clean_labels = [col.replace("_", " ").title() for col in df.columns]

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Pairwise Correlation Matrix: Demographic, Health & COVID-19 Metrics", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_file = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to {out_file}")

def plot_lagged_correlation():
    csv_path = OUTPUT_DIR / "lagged_correlation.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found. Skipping lagged correlation plot.")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["lag_days"], df["pearson_r"], marker="o", color="#d95f02", linewidth=2.5, markersize=7)
    plt.title("Epidemiological Latency: Time-Lagged Cross-Correlation\n(Daily Cases vs. Future Daily Deaths)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Time Lag Shift (Days)", fontsize=11)
    plt.ylabel("Pearson Correlation (r)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    max_idx = df["pearson_r"].idxmax()
    best_lag = df.loc[max_idx, "lag_days"]
    best_r = df.loc[max_idx, "pearson_r"]
    plt.annotate(
        f"Peak Latency Correlation: r={best_r:.3f} at {best_lag} days",
        xy=(best_lag, best_r),
        xytext=(best_lag - 5, best_r - 0.08),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontweight="bold"
    )

    plt.tight_layout()
    out_file = OUTPUT_DIR / "lagged_correlation_curve.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved lagged correlation chart to {out_file}")

def plot_vaccination_vs_cfr():
    csv_path = OUTPUT_DIR / "vaccination_vs_cfr.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} not found. Skipping vaccination vs CFR plot.")
        return

    df = pd.read_csv(csv_path)
    # Filter realistic CFR ranges to avoid noise/outliers
    df = df[(df["case_fatality_ratio_pct"] > 0) & (df["case_fatality_ratio_pct"] < 6)]

    plt.figure(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="people_fully_vaccinated_per_hundred",
        y="case_fatality_ratio_pct",
        scatter_kws={"alpha": 0.6, "color": "#1b9e77"},
        line_kws={"color": "#e7298a", "linewidth": 2}
    )
    plt.title("Correlation: Vaccination Coverage vs. Case Fatality Rate (CFR %)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("People Fully Vaccinated (Per 100 Population)", fontsize=11)
    plt.ylabel("Case Fatality Rate (%)", fontsize=11)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "vaccination_vs_cfr.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved vaccination vs CFR chart to {out_file}")

def generate_all_visualizations():
    logger.info("Generating all visualization figures...")
    plot_continent_growth()
    plot_top_peaks()
    plot_correlation_heatmap()
    plot_lagged_correlation()
    plot_vaccination_vs_cfr()
    logger.info("All plots generated successfully in data/output/.")

if __name__ == "__main__":
    generate_all_visualizations()

