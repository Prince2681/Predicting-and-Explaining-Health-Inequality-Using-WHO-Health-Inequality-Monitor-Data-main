"""
World Health Organization (WHO) Health Inequality Monitor
Analytical Dashboard and Disparity Risk Stratification System.
Engineered according to official WHO portal guidelines and visual identity.
"""
from pathlib import Path
import base64
import textwrap
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    BASE_DIR,
    DEFAULT_MODEL_PATH,
    IMAGES_DIR,
    WHO_REGIONS,
    INCOME_ORDER,
    RISK_TIERS,
    FEATURE_COLS
)
from src.data_loader import (
    load_raw_dataset,
    clean_and_engineer_features,
    prepare_train_test_data
)
from src.models import (
    train_classification_models,
    save_model,
    load_model,
    predict_risk_tier
)

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WHO Health Inequality Monitor",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Load external CSS
# ---------------------------------------------------------------------------
def load_css(css_file: str = "styles/style.css"):
    css_path = Path(css_file)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.html(f"<style>{f.read()}</style>")

load_css()

# ---------------------------------------------------------------------------
# Official WHO Emblem + Typography Base64 Loader
# ---------------------------------------------------------------------------
@st.cache_data
def get_logo_img_tag(width: int = 240) -> str:
    logo_path = Path("images/who_official_logo.svg")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" alt="World Health Organization" style="display:block;" />'
    return '<span style="font-size:1.3rem;font-weight:700;color:#0093D5;">World Health Organization</span>'


# ---------------------------------------------------------------------------
# Data & Model caching
# ---------------------------------------------------------------------------
@st.cache_data
def get_cached_data():
    raw_df = load_raw_dataset()
    model_df = clean_and_engineer_features(raw_df)
    return model_df


@st.cache_resource
def get_cached_model():
    if DEFAULT_MODEL_PATH.exists():
        return load_model(DEFAULT_MODEL_PATH)
    df = get_cached_data()
    splits = prepare_train_test_data(df)
    cls_out = train_classification_models(
        splits["X_train"], splits["y_train_cls"],
        splits["X_test"], splits["y_test_cls"]
    )
    rf = cls_out["models"]["Random Forest"]
    save_model(rf, DEFAULT_MODEL_PATH)
    return rf


# ---------------------------------------------------------------------------
# Top Brand Header (Clean WHO Design)
# ---------------------------------------------------------------------------
def render_brand_header():
    logo_tag = get_logo_img_tag(width=220)
    header_html = f"""<div class="who-brand-header">
<div class="who-brand-left">
{logo_tag}
</div>
<div class="who-brand-right">
<span class="who-portal-badge">Global Health Observatory Platform</span>
<div class="who-portal-name">Health Inequality Monitor</div>
<div class="who-portal-sub">Predicting &amp; Explaining Sex-Disaggregated Disparities</div>
</div>
</div>"""
    st.html(header_html)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # ----- Sidebar: Information & Metadata Only (No redundant navigation) -----
    logo_path = Path("images/who_official_logo.svg")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=180)

    st.sidebar.markdown(
        """<div style="font-size:0.75rem;font-weight:700;color:#1A3C6E;margin-top:0.4rem;margin-bottom:1rem;text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid #E2E8F0;padding-bottom:0.75rem;">
Health Inequality Monitor &bull; GHO
</div>""",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        """<div class="who-sidebar-card">
<strong>Portal Scope</strong><br>
Monitoring global sex-disaggregated health inequality across 198 member states and 131 SDG indicators.
</div>
<div class="who-sidebar-card">
<strong>Model Architecture</strong><br>
Optimized Random Forest Classifier with temporal holdout evaluation on 19,503 records (2020–2024).<br>
&bull; <strong>Out-of-Time Accuracy:</strong> 44.4%<br>
&bull; <strong>Macro ROC-AUC:</strong> 0.649
</div>
<div class="who-sidebar-card">
<strong>Risk Stratification Tiers</strong><br>
&bull; <strong>Low:</strong> |gap| &le; 0.90<br>
&bull; <strong>Medium:</strong> 0.90 &lt; |gap| &le; 4.20<br>
&bull; <strong>High:</strong> |gap| &gt; 4.20
</div>""",
        unsafe_allow_html=True
    )

    # ----- Top Brand Header -----
    render_brand_header()

    # ----- Functional Top Navigation Bar (Contains actual application content) -----
    tab_overview, tab_eda, tab_explain, tab_model = st.tabs([
        "Overview & Data Audit",
        "Exploratory Data Analysis",
        "Explainability & Validation",
        "Risk Stratification Tool"
    ])

    # ==================================================================
    # Tab 1: Overview & Data Audit
    # ==================================================================
    with tab_overview:
        with st.container(border=True):
            st.markdown("### Executive Summary & Global Monitoring Scope")
            st.caption("Standardized data audit and harmonization protocol across 198 WHO member states and 131 SDG health indicators.")

            # 4 Core KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harmonized Pairs", "79,448", help="Total matched female-male observation pairs")
            col2.metric("Countries Analyzed", "198", help="Global geographical member state coverage")
            col3.metric("SDG Indicators", "131", help="Sustainable Development Goal health indicators")
            col4.metric("Time Horizon", "1998 – 2024", help="26-year longitudinal monitoring horizon")

        st.html("""<div class="who-note">
<strong>Analytical Formulation:</strong> Health inequality across sex disaggregation strata is measured as:<br>
<code>Inequality Gap = Female Estimate &minus; Male Estimate</code><br>
A negative gap indicates higher female burden or male advantage (depending on indicator directionality), 
while a positive gap represents higher female outcome levels.
</div>""")

        with st.container(border=True):
            st.markdown("### Data Hygiene & Methodological Audit Protocol")
            table_markdown = textwrap.dedent("""\
            | Stage | Audit Focus | Methodological Implementation |
            |---|---|---|
            | **Grain Verification** | Unique record integrity | Verified grain uniqueness across Setting, Date, Indicator, and Subgroup. Exact duplicate records eliminated. |
            | **Value Validation** | Plausible rate bounds | Confirmed zero negative estimates across mortality, prevalence, and parity indicators. Pruned sparse features. |
            | **Regional Alignment** | Standardize geography | Mapped ISO3 country codes to the 6 official WHO Regional Offices (AFRO, AMRO, EMRO, EURO, SEARO, WPRO). |
            | **Risk Discretization** | Multi-class stratification | Established tercile risk tiers: **Low** (|gap| &le; 0.90), **Medium** (0.90 < |gap| &le; 4.20), **High** (|gap| > 4.20). |
            | **Temporal Holdout** | Prevent future data leakage | Historical records prior to 2020 used for training; 2020–2024 records reserved strictly for out-of-time evaluation. |
            """).strip()
            st.markdown(table_markdown)

        with st.container(border=True):
            st.markdown("### Enriched Observation Sample Explorer")
            try:
                df = get_cached_data()
                display_cols = [
                    "setting", "date", "indicator_abbr", "female_estimate",
                    "male_estimate", "inequality_gap", "whoreg6", "wbincome2025", "risk_category"
                ]
                st.dataframe(
                    df[display_cols].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.info(f"Observation cache initialization: {e}")

    # ==================================================================
    # Tab 2: Exploratory Data Analysis
    # ==================================================================
    with tab_eda:
        eda_options = [
            "Distribution & Longitudinal Trends",
            "Regional & Economic Stratification",
            "Statistical Hypotheses & Distributional Tests"
        ]
        eda_view = st.radio(
            "Select Analytical Perspective",
            eda_options,
            horizontal=True,
            label_visibility="collapsed"
        )

        st.write("")

        # Perspective 1: Distribution & Trends
        if eda_view == "Distribution & Longitudinal Trends":
            # Section A: Figure 1 Full-width clean card
            with st.container(border=True):
                st.markdown("### Figure 1: Inequality Gap Distribution (Middle 98% Density & Outliers)")
                st.caption("Distributional density trimmed to 1st–99th percentiles [-25.0, +25.0] to isolate core variance from extreme rate outliers.")

                stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
                stat_c1.metric("Trimmed Skewness", "-0.81", help="Near-symmetric central distribution mass")
                stat_c2.metric("Median Gap", "-0.30", help="Slight global female advantage across indicators")
                stat_c3.metric("Interquartile Range", "2.60", help="Middle 50% spread [-1.70 to +0.90]")
                stat_c4.metric("Extremes Observed", "[-100, +100]", help="Driven by 100-scale percentage and mortality rates")

                dist_img = IMAGES_DIR / "04_distribution.png"
                if dist_img.exists():
                    st.image(
                        str(dist_img),
                        caption="Figure 1: Probability density function and boxplot showing the centered mass of health inequality gaps.",
                        use_container_width=True
                    )

            # Section B: Balanced 2-Column Grid for Figure 2 and Figure 3
            col_left, col_right = st.columns([1, 1.25], gap="large")

            with col_left:
                with st.container(border=True):
                    st.markdown("### Figure 2: Disparity Directionality")
                    st.caption("Proportion of health gaps favoring females vs males globally.")

                    pie_img = IMAGES_DIR / "04_sign_pie.png"
                    if pie_img.exists():
                        st.image(
                            str(pie_img),
                            caption="Figure 2: Distribution of Disparity Sign",
                            use_container_width=True
                        )

                    direction_table = textwrap.dedent("""\
                    | Direction | Proportion | Description |
                    |---|---|---|
                    | **Female Advantage** (Gap < 0) | **61.8%** | Higher female outcomes / lower female mortality |
                    | **Male Advantage** (Gap > 0) | **34.1%** | Higher male outcomes / lower male mortality |
                    | **Gender Parity** (Gap = 0) | **4.1%** | Negligible sex differential |
                    """).strip()
                    st.markdown(direction_table)

            with col_right:
                with st.container(border=True):
                    st.markdown("### Figure 3: Longitudinal Convergence (1998–2023)")
                    st.caption("26-year historical trajectory of median inequality gaps across global health indicators.")

                    trend_img = IMAGES_DIR / "04_year_trend.png"
                    if trend_img.exists():
                        st.image(
                            str(trend_img),
                            caption="Figure 3: Global median inequality gap narrowed steadily from -1.50 in 1998 to -0.25 in 2023.",
                            use_container_width=True
                        )

                    st.html("""<div class="who-note">
<strong>Longitudinal Takeaways:</strong><br>
&bull; <strong>Steady Convergence:</strong> Median inequality gap has shrunk by &gt;80% over 26 years.<br>
&bull; <strong>Variance Compression:</strong> Inter-country disparity dispersions have tightened, reflecting progress in maternal and child health interventions.<br>
&bull; <strong>Residual Gaps:</strong> Gaps in adult NCDs and lifestyle indicators persist longer than infectious disease disparities.
</div>""")

        # Perspective 2: Regional & Economic Stratification
        elif eda_view == "Regional & Economic Stratification":
            with st.container(border=True):
                st.markdown("### Figure 4: Disparities by WHO Region & World Bank Income Group")
                st.caption("Cross-sectional comparison showing regional dispersion and income gradients.")

                reg_inc_img = IMAGES_DIR / "04_region_income.png"
                if reg_inc_img.exists():
                    st.image(
                        str(reg_inc_img),
                        caption="Figure 4: Middle 98% inequality gap distributions grouped by WHO Region (left) and World Bank Income Tier (right).",
                        use_container_width=True
                    )

                st.html("""<div class="who-note">
<strong>Key Regional &amp; Economic Insights:</strong><br>
&bull; <strong>Income Disparity Gradient:</strong> Absolute median gaps fall consistently as income increases: 
<strong>Low-Income (3.50)</strong> &rarr; <strong>Lower-Middle (2.40)</strong> &rarr; <strong>Upper-Middle (2.05)</strong> &rarr; <strong>High-Income (1.54)</strong>.<br>
&bull; <strong>Regional Heterogeneity:</strong> African and Eastern Mediterranean regions display wider variance spreads, whereas European member states show tight clustering around zero.<br>
&bull; <strong>Economic Buffer:</strong> Higher income tiers show stronger structural resilience against gender-based healthcare access disparities.
</div>""")

        # Perspective 3: Statistical Hypotheses & Tests
        elif eda_view == "Statistical Hypotheses & Distributional Tests":
            with st.container(border=True):
                st.markdown("### Statistical Hypotheses & Distributional Tests")
                test_table = textwrap.dedent("""\
                | Statistical Test | Null Hypothesis (H₀) | Test Statistic | p-Value | Outcome | Epidemiological Interpretation |
                |---|---|---|---|---|---|
                | **Shapiro-Wilk Test** | Inequality gap follows a Gaussian normal distribution | W = 0.0701 | p < 0.0001 | **Reject H₀** | Extreme kurtosis with multi-modal scales; parametric assumptions are invalid. |
                | **Mann-Whitney U Test** | Inequality gap distributions in High-income vs Low-income are identical | U = 7.15 &times; 10⁷ | p < 0.0001 | **Reject H₀** | Low-income nations experience statistically significant, structurally wider inequality gaps. |
                """).strip()
                st.markdown(test_table)

                st.caption(
                    "Methodological Conclusion: Raw skewness (-177.06) is an artifact of mixing non-comparable indicator scales "
                    "(rates per 100 vs rates per 100,000). Tercile risk stratification (|gap| <= 0.90, <= 4.20, > 4.20) solves this scale variance."
                )

    # ==================================================================
    # Tab 3: Explainability & Validation
    # ==================================================================
    with tab_explain:
        with st.container(border=True):
            st.markdown("### SHAP Feature Importance vs. Random Forest Gini Importance")
            st.caption("Marginal Shapley additive explanations (SHAP) compared to decision-tree impurity decrease (Gini).")

            col1, col2 = st.columns([1.3, 1], gap="medium")
            with col1:
                shap_img = IMAGES_DIR / "07_importance_comparison.png"
                if shap_img.exists():
                    st.image(str(shap_img), caption="Figure 5: Tree Gini Importance vs. Marginal SHAP Feature Attribution", use_container_width=True)
            with col2:
                st.html("""<div class="who-note">
<strong>Disparity Drivers (SHAP Attribution):</strong><br><br>
&bull; <strong>Indicator Scale (40%):</strong> Dominates prediction magnitude across rate denominators.<br><br>
&bull; <strong>Temporal Progression (20%):</strong> Captures longitudinal disparity narrowing across 1998–2024.<br><br>
&bull; <strong>Geographical &amp; Economic Context (31%):</strong> WHO Region (~17%) and Income Tier (~14%) form the baseline health capacity.<br><br>
&bull; <strong>Indicator Directionality (9%):</strong> Differentiates coverage goals from mortality indicators.
</div>""")

        with st.container(border=True):
            st.markdown("### Out-of-Time Model Validation (2020–2024 Holdout)")
            col_cm, col_insights = st.columns([1, 1.2], gap="medium")
            with col_cm:
                cm_img = IMAGES_DIR / "06_confusion_matrix.png"
                if cm_img.exists():
                    st.image(str(cm_img), caption="Figure 6: Confusion Matrix on 19,503 Unseen Records (2020–2024)", use_container_width=True)
            with col_insights:
                eval_md = textwrap.dedent("""\
                **Evaluation Highlights:**
                - **Strict Out-of-Time Holdout:** Validated on 19,503 observations recorded during 2020–2024 to test real future operational performance.
                - **Discriminative Capability:** Random Forest achieves **0.649 Macro ROC-AUC** and **44.4% Accuracy**, significantly outperforming baseline dummy guessing (33.0%).
                - **Balanced Macro F1 (0.430):** Confirms sensitive multi-class separation across Low, Medium, and High risk tiers without collapsing into majority-class bias.
                """).strip()
                st.markdown(eval_md)

    # ==================================================================
    # Tab 4: Risk Stratification Tool (Enhanced Model Building & Inference UI)
    # ==================================================================
    with tab_model:
        with st.container(border=True):
            st.markdown("### Interactive Disparity Risk Stratification System")
            st.caption(
                "Configure national and indicator attributes to estimate risk tier: "
                "**Low** (|gap| &le; 0.90), **Medium** (0.90 < |gap| &le; 4.20), or **High** (|gap| > 4.20)."
            )

            col_in1, col_in2 = st.columns(2, gap="large")
            with col_in1:
                st.markdown("#### Member State Profile")
                year = st.slider("Observation Year", min_value=1998, max_value=2030, value=2024, step=1)
                region = st.selectbox("WHO Region", WHO_REGIONS, index=3)
                income = st.selectbox("World Bank Income Group", INCOME_ORDER, index=3)
            with col_in2:
                st.markdown("#### Indicator Specifications")
                scale_options = [2, 100, 1000, 10000, 100000]
                scale_labels = {
                    2: "2 — Parity Index / Gender Ratio",
                    100: "100 — Percentage / Prevalence Rate (%)",
                    1000: "1,000 — Rate per 1,000 Live Births",
                    10000: "10,000 — Economic Output Units",
                    100000: "100,000 — Rate per 100,000 Population"
                }
                scale = st.selectbox(
                    "Indicator Scale",
                    scale_options,
                    index=1,
                    format_func=lambda x: scale_labels.get(x, str(x))
                )
                favourable = st.radio(
                    "Indicator Directionality",
                    [1, 0],
                    format_func=lambda x: "Favourable (higher is better, e.g. coverage)" if x == 1
                                else "Unfavourable (lower is better, e.g. mortality)"
                )

            st.write("")
            btn_col1, btn_col2 = st.columns([1, 2])
            with btn_col1:
                predict_clicked = st.button(" Run Stratification Assessment", use_container_width=True)

            if predict_clicked:
                try:
                    model = get_cached_model()
                    res = predict_risk_tier(
                        model=model,
                        date=year,
                        region=region,
                        income_group=income,
                        indicator_scale=scale,
                        favourable_indicator=favourable
                    )

                    tier = res["predicted_tier"]
                    probs = res["probabilities"]

                    st.write("")
                    st.markdown("---")
                    st.markdown("### Stratification Assessment Report")

                    # WHO Official Severity Banner
                    if tier == "Low":
                        st.html("""<div class="who-risk-banner-low">
<strong style="font-size:1.15rem;"> Stratified Assessment: LOW DISPARITY RISK (|gap| &le; 0.90)</strong><br>
Minimal sex differential detected for this indicator configuration. Health outcomes between females and males are near parity. Routine monitoring advised.
</div>""")
                    elif tier == "Medium":
                        st.html("""<div class="who-risk-banner-med">
<strong style="font-size:1.15rem;"> Stratified Assessment: MEDIUM DISPARITY RISK (0.90 &lt; |gap| &le; 4.20)</strong><br>
Noticeable inequality gap predicted. Sex-stratified healthcare access or outcome disparity requires focused targeted programmatic attention.
</div>""")
                    else:
                        st.html("""<div class="who-risk-banner-high">
<strong style="font-size:1.15rem;"> Stratified Assessment: HIGH DISPARITY RISK (|gap| &gt; 4.20)</strong><br>
Critical disparity predicted. Severe gender-based health gap likely under this demographic and economic configuration. Priority policy intervention recommended.
</div>""")

                    # Posterior Probability Distribution
                    st.markdown("####  Posterior Probability Distribution")
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric("Low Risk Probability", f"{probs['Low'] * 100:.1f}%")
                    p_col2.metric("Medium Risk Probability", f"{probs['Medium'] * 100:.1f}%")
                    p_col3.metric("High Risk Probability", f"{probs['High'] * 100:.1f}%")

                    prob_df = pd.DataFrame(
                        list(probs.items()),
                        columns=["Risk Tier", "Probability"]
                    )
                    st.bar_chart(prob_df.set_index("Risk Tier"), height=240)

                    # Dynamic Policy & Feature Impact Takeaway
                    with st.container(border=True):
                        st.markdown("#### Key Drivers Behind This Prediction")
                        st.markdown(f"""
                        - **Indicator Scale ({scale}):** Accounts for ~40% of baseline dispersion magnitude.
                        - **Geographic Context ({region} Region):** Contributes regional health infrastructure baseline (~17% weight).
                        - **Economic Tier ({income}):** Determines domestic health expenditure capacity (~14% weight).
                        - **Observation Horizon ({year}):** Incorporates the 26-year longitudinal global convergence trend.
                        """)

                except Exception as e:
                    st.error(f"Error computing stratification report: {e}")

        # Benchmark Comparison Table
        with st.container(border=True):
            st.markdown("### Machine Learning Model Benchmark Comparison")
            st.caption("Rigorous out-of-time evaluation across 19,503 holdout records (2020–2024).")
            benchmark_table = textwrap.dedent("""\
            | Model Architecture | Test Accuracy | Macro F1 | Macro ROC-AUC | Primary Advantage |
            |---|---|---|---|---|
            | **Random Forest (Primary)** | **44.4%** | **0.430** | **0.649** | **Highest discriminatory power; robust against outlier skewness** |
            | XGBoost Classifier | 44.3% | 0.431 | 0.648 | High gradient boosting precision; competitive F1 score |
            | Decision Tree | 44.2% | 0.423 | 0.647 | Fast interpretable rule tree |
            | Logistic Regression | 37.6% | 0.368 | 0.575 | Linear baseline; underfits multi-scale variance |
            | Stratified Dummy Baseline | 33.0% | 0.330 | 0.498 | Random chance benchmark |
            """).strip()
            st.markdown(benchmark_table)

    # ----- Global WHO Footer -----
    st.html("""<div class="who-footer-banner">
<div>
&copy; World Health Organization 2026. All rights reserved.
Health Inequality Monitor &mdash; Global Health Observatory (GHO).
</div>
<div>
<a href="https://www.who.int/data/gho/health-equity-monitor" target="_blank">Health Equity Monitor Official Portal</a>
</div>
</div>""")


if __name__ == "__main__":
    main()
