"""
Exploratory Data Analysis (EDA) and Statistical Testing Module (Module 4).
Calculates summary statistics, runs hypothesis tests, and generates visualizations.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / Docker execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import IMAGES_DIR, LOW_RISK_CUTOFF, MED_RISK_CUTOFF, INCOME_ORDER


def compute_gap_statistics(df: pd.DataFrame) -> dict:
    """Computes descriptive and distribution statistics for inequality gap."""
    gap = df["inequality_gap"]
    p01 = gap.quantile(0.01)
    p99 = gap.quantile(0.99)
    trimmed = df[(gap >= p01) & (gap <= p99)]

    neg_count = (gap < 0).sum()
    pos_count = (gap > 0).sum()
    zero_count = (gap == 0).sum()
    total = len(gap)

    return {
        "count": total,
        "mean": float(gap.mean()),
        "std": float(gap.std()),
        "median": float(gap.median()),
        "min": float(gap.min()),
        "max": float(gap.max()),
        "raw_skewness": float(gap.skew()),
        "trimmed_skewness": float(trimmed["inequality_gap"].skew()),
        "negative_pct": float(neg_count / total * 100),
        "positive_pct": float(pos_count / total * 100),
        "zero_pct": float(zero_count / total * 100),
        "p01": float(p01),
        "p99": float(p99)
    }


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Performs Shapiro-Wilk normality test and Mann-Whitney U test
    between High-income and Low-income settings.
    """
    results = {}

    # 1. Shapiro-Wilk Normality Test (Sampled to max 5000 for validity)
    sample = df["inequality_gap"].sample(min(5000, len(df)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    results["shapiro_wilk"] = {
        "statistic": float(sw_stat),
        "p_value": float(sw_p),
        "is_normal": bool(sw_p > 0.05)
    }

    # 2. Mann-Whitney U Test: High vs Low Income
    if "wbincome2025" in df.columns:
        high = df[df["wbincome2025"] == "High-income"]["inequality_gap"].dropna()
        low = df[df["wbincome2025"] == "Low-income"]["inequality_gap"].dropna()
        if len(high) > 0 and len(low) > 0:
            mwu_stat, mwu_p = stats.mannwhitneyu(high, low, alternative="two-sided")
            results["mann_whitney_u"] = {
                "high_count": len(high),
                "low_count": len(low),
                "statistic": float(mwu_stat),
                "p_value": float(mwu_p),
                "is_significant": bool(mwu_p < 0.05)
            }

    return results


def generate_eda_plots(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> dict:
    """Generates and saves all EDA charts (Module 4)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    # 1. Distribution & Outliers (04_distribution.png)
    p01 = df["inequality_gap"].quantile(0.01)
    p99 = df["inequality_gap"].quantile(0.99)
    trimmed = df[(df["inequality_gap"] >= p01) & (df["inequality_gap"] <= p99)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(trimmed["inequality_gap"], bins=60, color="#4A90D9", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Inequality Gap Distribution\n(Middle 98% trimmed)")
    axes[0].set_xlabel("inequality_gap")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(df["inequality_gap"], vert=True)
    axes[1].set_title("Inequality Gap Boxplot\n(Full data with outliers)")
    axes[1].set_ylabel("inequality_gap")
    plt.tight_layout()
    dist_path = output_dir / "04_distribution.png"
    plt.savefig(dist_path, dpi=150)
    plt.close(fig)
    generated_files["distribution"] = dist_path

    # 2. Direction of Gap (04_sign_pie.png)
    sign_counts = pd.Series({
        "Negative\n(Female < Male)": (df["inequality_gap"] < 0).sum(),
        "Positive\n(Female > Male)": (df["inequality_gap"] > 0).sum(),
        "Zero": (df["inequality_gap"] == 0).sum(),
    })
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sign_counts,
        labels=sign_counts.index,
        autopct="%1.1f%%",
        colors=["#E8584A", "#4A90D9", "#95A5A6"],
        startangle=90
    )
    ax.set_title("Direction of Health Inequality Gap")
    plt.tight_layout()
    sign_path = output_dir / "04_sign_pie.png"
    plt.savefig(sign_path, dpi=150)
    plt.close(fig)
    generated_files["sign_pie"] = sign_path

    # 3. Yearly Trend (04_year_trend.png)
    yearly = df[df["date"] < 2024].groupby("date")["inequality_gap"].median()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values, marker="o", color="#4A90D9", linewidth=2)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Median Inequality Gap by Year (1998-2023)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median inequality_gap")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    trend_path = output_dir / "04_year_trend.png"
    plt.savefig(trend_path, dpi=150)
    plt.close(fig)
    generated_files["year_trend"] = trend_path

    # 4. Region & Income Breakdown (04_region_income.png)
    if "whoreg6" in df.columns and "wbincome2025" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        region_order = (
            df.groupby("whoreg6")["inequality_gap"].median().sort_values().index
        )
        sns.boxplot(
            data=trimmed,
            x="whoreg6",
            y="inequality_gap",
            hue="whoreg6",
            legend=False,
            order=region_order,
            ax=axes[0],
            palette="Blues"
        )
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_title("Inequality Gap by WHO Region (Middle 98%)")
        axes[0].tick_params(axis="x", rotation=30)

        sns.boxplot(
            data=trimmed,
            x="wbincome2025",
            y="inequality_gap",
            hue="wbincome2025",
            legend=False,
            order=INCOME_ORDER,
            ax=axes[1],
            palette="Oranges"
        )
        axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_title("Inequality Gap by Income Group (Middle 98%)")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        reg_inc_path = output_dir / "04_region_income.png"
        plt.savefig(reg_inc_path, dpi=150)
        plt.close(fig)
        generated_files["region_income"] = reg_inc_path

    return generated_files
