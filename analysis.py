#!/usr/bin/env python3
"""
Analysis script for:
"Combined Noise Exposure and Hearing Difficulty as Independent Predictors
of Tinnitus-Related Sleep Disturbance: A Nationally Representative Analysis
of U.S. Adults"

Data source: 2014 National Health Interview Survey (NHIS)
Target journal: Noise and Health

Requirements:
    - pandas, numpy, scipy, statsmodels, matplotlib
    - NHIS 2014 files: samadult.dat, personsx.dat
      Download from: https://archive.cdc.gov/www_cdc_gov/nchs/nhis/nhis_2014_data_release.htm
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = '.'  # Directory containing samadult.dat and personsx.dat
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMADULT_FILE = os.path.join(DATA_DIR, 'samadult.dat')
PERSONSX_FILE = os.path.join(DATA_DIR, 'personsx.dat')


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_nhis_data():
    """Load and merge NHIS 2014 Sample Adult and Person files."""

    print("Loading Person file (for education)...")
    person_specs = [(6, 12), (15, 17), (17, 19), (784, 786)]
    person_names = ['HHX', 'FMX', 'FPX', 'EDUC1']
    person = pd.read_fwf(
        PERSONSX_FILE, colspecs=person_specs,
        names=person_names, header=None
    )
    for c in person.columns:
        person[c] = pd.to_numeric(person[c], errors='coerce')

    print("Loading Sample Adult file...")
    sa_specs = {
        'HHX': (6, 12), 'FMX': (15, 17), 'FPX': (17, 19),
        'WTFA_SA': (26, 32), 'STRAT_P': (33, 36), 'PSU_P': (36, 38),
        'SEX': (38, 39), 'HISPAN_I': (39, 41), 'RACERPI2': (41, 43),
        'AGE_P': (47, 49), 'AHEARST2': (295, 296),
        'HRTIN': (353, 354), 'HRTINSLP': (358, 359), 'HRTINPRO': (359, 360),
        'HRFIRE': (378, 379), 'HRWKVLNS': (385, 386), 'HRWKLNS': (386, 387),
        'HRWKVLP1': (390, 391), 'HRWKLP1': (395, 396),
        'HRLSVLNS': (397, 398), 'HRLSP1': (420, 421),
        'SMKSTAT2': (747, 748), 'ASISLEEP': (1056, 1058),
    }
    names = list(sa_specs.keys())
    df = pd.read_fwf(
        SAMADULT_FILE,
        colspecs=[sa_specs[k] for k in names],
        names=names, header=None
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print("Merging education from Person file...")
    df = df.merge(
        person[['HHX', 'FMX', 'FPX', 'EDUC1']],
        on=['HHX', 'FMX', 'FPX'], how='left'
    )

    return df


# ============================================================
# 2. DATA CLEANING AND RECODING
# ============================================================
def recode_variables(df):
    """Apply NHIS missing codes and create analytic variables."""

    # Restrict to adults 18+
    df = df[df['AGE_P'] >= 18].copy()

    # --- Missing value codes ---
    single_digit_missing = [7, 8, 9]
    for v in ['HRTIN', 'HRTINSLP', 'HRTINPRO', 'HRFIRE', 'HRWKVLNS',
              'HRWKLNS', 'HRWKVLP1', 'HRWKLP1', 'HRLSVLNS', 'HRLSP1',
              'SMKSTAT2', 'SEX', 'AHEARST2']:
        df.loc[df[v].isin(single_digit_missing), v] = np.nan

    for v in ['ASISLEEP', 'HISPAN_I', 'RACERPI2']:
        df.loc[df[v].isin([97, 98, 99]), v] = np.nan

    df.loc[df['ASISLEEP'] > 24, 'ASISLEEP'] = np.nan
    df.loc[df['EDUC1'].isin([96, 97, 98, 99]), 'EDUC1'] = np.nan

    # --- Education (reference = Bachelor+) ---
    df['education'] = np.nan
    df.loc[df['EDUC1'].between(18, 21), 'education'] = 0   # Bachelor+ (ref)
    df.loc[df['EDUC1'].between(0, 14), 'education'] = 1    # < High school
    df.loc[df['EDUC1'] == 15, 'education'] = 2              # HS graduate
    df.loc[df['EDUC1'].isin([16, 17]), 'education'] = 3     # Some college

    # --- Hearing difficulty (binary) ---
    # AHEARST2: 1-2 = good hearing, 3-6 = any difficulty
    df['hear_diff'] = np.where(
        df['AHEARST2'].isin([1, 2]), 0,
        np.where(df['AHEARST2'].isin([3, 4, 5, 6]), 1, np.nan)
    )

    # --- Outcomes ---
    df['tinnitus'] = np.where(df['HRTIN'] == 1, 1,
                    np.where(df['HRTIN'] == 2, 0, np.nan))

    df['tin_sleep'] = np.where(df['HRTINSLP'] == 1, 1,
                     np.where(df['HRTINSLP'] == 2, 0, np.nan))

    df['tin_problem'] = np.where(
        df['HRTINPRO'].isin([3, 4, 5]), 1,
        np.where(df['HRTINPRO'].isin([1, 2]), 0, np.nan)
    )

    df['short_sleep'] = np.where(
        df['ASISLEEP'] < 7, 1,
        np.where(df['ASISLEEP'] >= 7, 0, np.nan)
    )

    # --- Noise exposure ---
    df['occ_noise'] = np.where(
        (df['HRWKVLNS'] == 1) | (df['HRWKLNS'] == 1), 1,
        np.where((df['HRWKVLNS'] == 2) & (df['HRWKLNS'] == 2), 0, np.nan)
    )

    df['lei_noise'] = np.where(
        df['HRLSVLNS'] == 1, 1,
        np.where(df['HRLSVLNS'] == 2, 0, np.nan)
    )

    # 4-level exposure group
    conditions = [
        (df['occ_noise'] == 0) & (df['lei_noise'] == 0),
        (df['occ_noise'] == 1) & (df['lei_noise'] == 0),
        (df['occ_noise'] == 0) & (df['lei_noise'] == 1),
        (df['occ_noise'] == 1) & (df['lei_noise'] == 1),
    ]
    df['exp_group'] = np.select(conditions, [0, 1, 2, 3], default=np.nan)

    # --- Hearing protection ---
    def recode_protection(col):
        return np.where(col.isin([1, 2]), 1,
               np.where(col.isin([3, 4, 5]), 0, np.nan))

    df['work_prot'] = np.where(
        (recode_protection(df['HRWKVLP1']) == 1) |
        (recode_protection(df['HRWKLP1']) == 1), 1,
        np.where(df['occ_noise'] == 1, 0, np.nan)
    )

    df['firearm'] = np.where(df['HRFIRE'] == 1, 1,
                   np.where(df['HRFIRE'] == 2, 0, np.nan))

    # --- Covariates ---
    df['female'] = np.where(df['SEX'] == 2, 1,
                  np.where(df['SEX'] == 1, 0, np.nan))

    # Race/ethnicity: HISPAN_I 0-11=Hispanic, 12=Not Hispanic
    df['race_eth'] = np.nan
    df.loc[df['HISPAN_I'].between(0, 11), 'race_eth'] = 3  # Hispanic
    df.loc[(df['HISPAN_I'] == 12) & (df['RACERPI2'] == 1), 'race_eth'] = 0  # NH White
    df.loc[(df['HISPAN_I'] == 12) & (df['RACERPI2'] == 2), 'race_eth'] = 1  # NH Black
    df.loc[
        (df['HISPAN_I'] == 12) &
        (~df['RACERPI2'].isin([1, 2])) &
        (df['RACERPI2'].notna()),
        'race_eth'
    ] = 2  # NH Other

    # Smoking: 1-2=current, 3=former, 4=never
    df['smoking'] = np.where(
        df['SMKSTAT2'].isin([1, 2]), 2,
        np.where(df['SMKSTAT2'] == 3, 1,
        np.where(df['SMKSTAT2'] == 4, 0, np.nan))
    )

    # --- Dummy variables ---
    for i in [1, 2, 3]:
        df[f'exp_{i}'] = (df['exp_group'] == i).astype(float)
        df[f'race_{i}'] = (df['race_eth'] == i).astype(float)
        df[f'educ_{i}'] = (df['education'] == i).astype(float)
    for i in [1, 2]:
        df[f'smk_{i}'] = (df['smoking'] == i).astype(float)

    df['age_c'] = df['AGE_P']
    df['exp_ordinal'] = df['exp_group']

    return df


# ============================================================
# 3. SURVEY-WEIGHTED LOGISTIC REGRESSION
#    Taylor series linearization for design-based SEs
# ============================================================
COVARIATE_SET = [
    'age_c', 'female',
    'race_1', 'race_2', 'race_3',
    'educ_1', 'educ_2', 'educ_3',
    'smk_1', 'smk_2',
    'hear_diff'
]


def survey_logistic(data, outcome, predictors, weight_col='WTFA_SA',
                    strat_col='STRAT_P', psu_col='PSU_P', model_name='Model'):
    """
    Weighted logistic regression with Taylor series linearization
    for stratified two-stage cluster design (NHIS pseudo-strata/PSUs).

    Parameters
    ----------
    data : DataFrame
    outcome : str
    predictors : list of str
    weight_col, strat_col, psu_col : str
    model_name : str

    Returns
    -------
    DataFrame with OR, CI_low, CI_high, p_value for each predictor
    """
    cols_needed = predictors + [outcome, weight_col, strat_col, psu_col]
    ad = data[cols_needed].dropna().copy()

    y = ad[outcome].values
    X = sm.add_constant(ad[predictors]).values
    w = ad[weight_col].values
    strat = ad[strat_col].values.astype(int)
    psu = ad[psu_col].values.astype(int)
    n = len(ad)
    p_dim = X.shape[1]

    # Point estimates via weighted GLM
    w_norm = w / w.mean()
    result = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w_norm).fit()
    beta = result.params

    # Predicted probabilities and score contributions
    p_hat = 1.0 / (1.0 + np.exp(-X @ beta))
    U = w.reshape(-1, 1) * (y - p_hat).reshape(-1, 1) * X

    # Information (bread) matrix
    W_diag = w * p_hat * (1.0 - p_hat)
    H = (X.T * W_diag) @ X
    H_inv = np.linalg.inv(H)

    # Taylor linearization (meat) matrix
    V = np.zeros((p_dim, p_dim))
    for h in np.unique(strat):
        mask_h = strat == h
        psus_in_h = np.unique(psu[mask_h])
        n_h = len(psus_in_h)
        if n_h < 2:
            continue
        psu_totals = np.array([
            U[mask_h & (psu == a)].sum(axis=0) for a in psus_in_h
        ])
        dev = psu_totals - psu_totals.mean(axis=0)
        V += (n_h / (n_h - 1.0)) * (dev.T @ dev)

    # Sandwich covariance
    cov_beta = H_inv @ V @ H_inv
    se_design = np.sqrt(np.diag(cov_beta))

    # Inference
    z_vals = beta / se_design
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_vals)))

    var_names = ['const'] + predictors
    ordf = pd.DataFrame({
        'OR': np.exp(beta),
        'CI_low': np.exp(beta - 1.96 * se_design),
        'CI_high': np.exp(beta + 1.96 * se_design),
        'p_value': p_values,
        'SE_design': se_design,
    }, index=var_names)

    # Print summary
    events = int(y.sum())
    print(f"\n  {model_name} (n={n:,}, events={events:,})")
    print(f"  {'Variable':<25s} {'OR':>7s} {'95% CI':>18s} {'p':>9s}")
    print(f"  {'-' * 65}")
    for idx, row in ordf.iterrows():
        if idx == 'const':
            continue
        sig = ('***' if row['p_value'] < 0.001
               else '**' if row['p_value'] < 0.01
               else '*' if row['p_value'] < 0.05
               else '')
        print(f"  {idx:<25s} {row['OR']:>7.2f} "
              f"({row['CI_low']:>5.2f} - {row['CI_high']:<5.2f}) "
              f"{row['p_value']:>8.4f} {sig}")

    return ordf


# ============================================================
# 4. WEIGHTED DESCRIPTIVES
# ============================================================
def weighted_prevalence(data, var, weight='WTFA_SA'):
    """Compute weighted percentage distribution."""
    valid = data[[var, weight]].dropna()
    grouped = valid.groupby(var)[weight].sum()
    return (grouped / grouped.sum() * 100).round(1)


def weighted_mean(data, var, weight='WTFA_SA'):
    """Compute weighted mean."""
    valid = data[[var, weight]].dropna()
    return np.average(valid[var], weights=valid[weight])


# ============================================================
# 5. MAIN ANALYSIS
# ============================================================
def main():
    print("=" * 65)
    print("NHIS 2014: Noise, Hearing Difficulty, and Tinnitus Sleep Disturbance")
    print("=" * 65)

    # Load and prepare data
    df = load_nhis_data()
    df = recode_variables(df)

    # Define analytic samples
    sample_criteria = (
        (df['exp_group'].notna()) &
        (df['WTFA_SA'] > 0) &
        (df['education'].notna()) &
        (df['hear_diff'].notna()) &
        (df['STRAT_P'].notna()) &
        (df['PSU_P'].notna())
    )

    s1 = df[
        sample_criteria &
        (df['tinnitus'] == 1) &
        (df['tin_sleep'].notna())
    ].copy()

    s2 = df[
        sample_criteria &
        (df['short_sleep'].notna())
    ].copy()

    print(f"\n  Tinnitus subsample (S1): n = {len(s1):,}")
    print(f"  Full adult sample (S2):  n = {len(s2):,}")
    print(f"  Hearing difficulty prevalence (S2): {s2['hear_diff'].mean()*100:.1f}%")
    print(f"  Hearing difficulty among tinnitus (S1): {s1['hear_diff'].mean()*100:.1f}%")

    # --------------------------------------------------------
    # TABLE 1: Characteristics by exposure group
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("TABLE 1: Sample Characteristics by Noise Exposure Group")
    print("=" * 65)

    exp_labels = {0: 'No Noise', 1: 'Occup Only', 2: 'Leisure Only', 3: 'Both'}
    educ_labels = {0: 'Bachelor+', 1: '<HS', 2: 'HS grad', 3: 'Some college'}

    for grp in [0, 1, 2, 3]:
        sub = s2[s2['exp_group'] == grp]
        n = len(sub)
        wn = sub['WTFA_SA'].sum()
        print(f"\n  {exp_labels[grp]} (n={n:,}, weighted N={wn:,.0f}):")
        print(f"    Female: {weighted_prevalence(sub, 'female').get(1, 0):.1f}%")
        print(f"    Mean age: {weighted_mean(sub, 'AGE_P'):.1f}")
        print(f"    Hearing difficulty: {weighted_prevalence(sub, 'hear_diff').get(1, 0):.1f}%")
        print(f"    Tinnitus: {weighted_prevalence(sub, 'tinnitus').get(1, 0):.1f}%")
        print(f"    Short sleep: {weighted_prevalence(sub, 'short_sleep').get(1, 0):.1f}%")
        print(f"    Current smoker: {weighted_prevalence(sub, 'smoking').get(2, 0):.1f}%")
        print(f"    Firearm use: {weighted_prevalence(sub, 'firearm').get(1, 0):.1f}%")

    # --------------------------------------------------------
    # TABLE 2: Weighted prevalence of outcomes
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("TABLE 2: Weighted Prevalence of Key Outcomes")
    print("=" * 65)

    tw = s2['WTFA_SA'].sum()
    print(f"  Overall tinnitus: {s2.loc[s2['tinnitus']==1, 'WTFA_SA'].sum()/tw*100:.1f}%")
    print(f"  Overall short sleep: {s2.loc[s2['short_sleep']==1, 'WTFA_SA'].sum()/tw*100:.1f}%")

    for grp in [0, 1, 2, 3]:
        sub = s2[s2['exp_group'] == grp]
        t = sub['WTFA_SA'].sum()
        tin = sub.loc[sub['tinnitus'] == 1, 'WTFA_SA'].sum()
        ss = sub.loc[sub['short_sleep'] == 1, 'WTFA_SA'].sum()
        print(f"  {exp_labels[grp]}: Tinnitus={tin/t*100:.1f}%, Short sleep={ss/t*100:.1f}%")

    print("\n  Among tinnitus subsample:")
    for grp in [0, 1, 2, 3]:
        sub = s1[s1['exp_group'] == grp]
        if len(sub) > 0:
            t = sub['WTFA_SA'].sum()
            ts = sub.loc[sub['tin_sleep'] == 1, 'WTFA_SA'].sum()
            print(f"    {exp_labels[grp]} (n={len(sub)}): "
                  f"Sleep disturbance = {ts/t*100:.1f}%")

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION MODELS (hearing-adjusted, design-based SEs)")
    print("=" * 65)

    # Model 1: Tinnitus sleep disturbance
    print("\n--- Model 1: Tinnitus-Related Sleep Disturbance ---")
    m1 = survey_logistic(
        s1, 'tin_sleep',
        ['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET,
        model_name='Model 1 (Adjusted)'
    )

    # Model 2: Tinnitus problem severity
    print("\n--- Model 2: Tinnitus Problem Severity (Moderate+) ---")
    m2 = survey_logistic(
        s1, 'tin_problem',
        ['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET,
        model_name='Model 2 (Adjusted)'
    )

    # Model 3: Short sleep
    print("\n--- Model 3: Short Sleep (<7h) ---")
    m3 = survey_logistic(
        s2, 'short_sleep',
        ['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET,
        model_name='Model 3b (Adjusted)'
    )

    # Model 3c: Short sleep + tinnitus (for decomposition)
    m3c = survey_logistic(
        s2, 'short_sleep',
        ['exp_1', 'exp_2', 'exp_3', 'tinnitus'] + COVARIATE_SET,
        model_name='Model 3c (+ Tinnitus)'
    )

    # Model 4 (Exploratory): Hearing protection
    print("\n--- Model 4 (Exploratory): Work Protection ---")
    s1_occ = s1[(s1['occ_noise'] == 1) & (s1['work_prot'].notna())].copy()
    print(f"  Subsample: n = {len(s1_occ)}")
    m4 = survey_logistic(
        s1_occ, 'tin_sleep',
        ['work_prot'] + COVARIATE_SET,
        model_name='Model 4 (Exploratory)'
    )

    # --------------------------------------------------------
    # P-FOR-TREND
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("P-FOR-TREND")
    print("=" * 65)

    for samp, outcome, label in [
        (s1, 'tin_sleep', 'Tinnitus sleep disturbance'),
        (s2, 'short_sleep', 'Short sleep'),
    ]:
        r = survey_logistic(
            samp, outcome,
            ['exp_ordinal'] + COVARIATE_SET,
            model_name=f'Trend: {label}'
        )
        print(f"  >>> {label}: OR/unit = {r.loc['exp_ordinal', 'OR']:.3f}, "
              f"p-trend = {r.loc['exp_ordinal', 'p_value']:.2e}")

    # --------------------------------------------------------
    # DECOMPOSITION ANALYSIS (KHB method)
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("DECOMPOSITION ANALYSIS (KHB method)")
    print("=" * 65)

    decomp_cols = (['exp_1', 'exp_2', 'exp_3', 'tinnitus'] +
                   COVARIATE_SET +
                   ['short_sleep', 'WTFA_SA', 'STRAT_P', 'PSU_P'])
    ad = s2[decomp_cols].dropna().copy()
    w_n = ad['WTFA_SA'].values / ad['WTFA_SA'].mean()

    # Step 1: Regress mediator on exposure + confounders
    X_conf = sm.add_constant(ad[['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET])
    med_model = sm.WLS(ad['tinnitus'], X_conf, weights=w_n).fit()
    ad['tin_resid'] = (ad['tinnitus'] - med_model.fittedvalues).values

    # Step 2: Full model (direct effect)
    full_preds = ['exp_1', 'exp_2', 'exp_3', 'tinnitus'] + COVARIATE_SET
    X_full = sm.add_constant(ad[full_preds]).values
    full_fit = sm.GLM(
        ad['short_sleep'].values, X_full,
        family=sm.families.Binomial(), freq_weights=w_n
    ).fit()

    # Step 3: KHB-reduced model (total effect on same scale)
    khb_preds = ['exp_1', 'exp_2', 'exp_3', 'tin_resid'] + COVARIATE_SET
    X_khb = sm.add_constant(ad[khb_preds]).values
    khb_fit = sm.GLM(
        ad['short_sleep'].values, X_khb,
        family=sm.families.Binomial(), freq_weights=w_n
    ).fit()

    print(f"\n  {'Exposure':<20s} {'Total(KHB)':>10s} {'Direct':>10s} "
          f"{'Indirect':>10s} {'%Decomp':>8s}")
    print(f"  {'-' * 62}")
    for i, (v, l) in enumerate(zip(
        ['exp_1', 'exp_2', 'exp_3'],
        ['Occup only', 'Leisure only', 'Both']
    )):
        total = khb_fit.params[i + 1]
        direct = full_fit.params[i + 1]
        indirect = total - direct
        pct = (indirect / total * 100) if total != 0 else 0
        print(f"  {l:<20s} {total:>10.4f} {direct:>10.4f} "
              f"{indirect:>10.4f} {pct:>7.1f}%")

    # --------------------------------------------------------
    # ESTIMATED PAF (conditional on causal assumption)
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("ESTIMATED PAF (conditional on causal assumption)")
    print("=" * 65)

    total_cases_w = s1.loc[s1['tin_sleep'] == 1, 'WTFA_SA'].sum()
    paf_total = 0
    for i in [1, 2, 3]:
        cases_w = s1.loc[
            (s1['exp_group'] == i) & (s1['tin_sleep'] == 1), 'WTFA_SA'
        ].sum()
        pd_i = cases_w / total_cases_w
        or_i = m1.loc[f'exp_{i}', 'OR']
        paf_i = pd_i * (or_i - 1) / or_i
        paf_total += paf_i
        print(f"  exp_{i}: pd={pd_i:.3f}, aOR={or_i:.2f}, PAF_i={paf_i:.4f}")
    print(f"  Estimated total PAF: {paf_total * 100:.1f}%")

    # --------------------------------------------------------
    # SENSITIVITY ANALYSES
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("SENSITIVITY ANALYSES")
    print("=" * 65)

    # Sensitivity 1: Very loud occupational only
    s1['occ_vloud'] = np.where(
        s1['HRWKVLNS'] == 1, 1,
        np.where(s1['HRWKVLNS'] == 2, 0, np.nan)
    )
    survey_logistic(
        s1, 'tin_sleep',
        ['occ_vloud', 'lei_noise'] + COVARIATE_SET,
        model_name='Sensitivity 1: Very loud occupational only'
    )

    # Sensitivity 2: Stratified by firearm use
    s1_nogun = s1[s1['firearm'] == 0].copy()
    s1_gun = s1[s1['firearm'] == 1].copy()
    print(f"\n  No firearm: n={len(s1_nogun)}, Firearm: n={len(s1_gun)}")

    survey_logistic(
        s1_nogun, 'tin_sleep',
        ['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET,
        model_name='Sensitivity 2a: No firearm stratum'
    )
    survey_logistic(
        s1_gun, 'tin_sleep',
        ['exp_1', 'exp_2', 'exp_3'] + COVARIATE_SET,
        model_name='Sensitivity 2b: Firearm stratum'
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------
    print("\nGenerating figures...")
    generate_figures(s1, s2, m1, m2, m3)

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE. Output saved to:", OUTPUT_DIR)
    print("=" * 65)


# ============================================================
# 6. FIGURE GENERATION
# ============================================================
def recode_prot_for_fig(col):
    return np.where(col.isin([1, 2]), 1, np.where(col.isin([3, 4, 5]), 0, np.nan))


def generate_figures(s1, s2, m1, m2, m3):
    """Generate manuscript figures."""

    # --- Figure 1: Forest Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vars = ['exp_1', 'exp_2', 'exp_3']
    labels = [
        'Occupational noise\nonly',
        'Leisure noise\nonly',
        'Both occupational\n+ leisure',
    ]

    model_configs = [
        ('Tinnitus sleep disturbance', m1, '#1f77b4', 's', 0.25),
        ('Tinnitus severity (mod+)', m2, '#ff7f0e', '^', 0),
        ('Short sleep (<7h)', m3, '#d62728', 'D', -0.25),
    ]

    for label, mdata, color, marker, offset in model_configs:
        for i, v in enumerate(plot_vars):
            o = mdata.loc[v, 'OR']
            lo = mdata.loc[v, 'CI_low']
            hi = mdata.loc[v, 'CI_high']
            ax.errorbar(
                o, i + offset,
                xerr=[[o - lo], [hi - o]],
                fmt=marker, color=color,
                markersize=9, capsize=4, linewidth=2,
                label=label if i == 0 else ''
            )

    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(
        'Adjusted OR (95% CI)\nDesign-based standard errors',
        fontsize=11
    )
    ax.set_title(
        'Figure 1. Adjusted ORs by Noise Exposure Group\n'
        '(Reference: No noise; adjusted for age, sex, race/ethnicity, '
        'education, smoking, hearing difficulty)',
        fontsize=12
    )
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.invert_yaxis()
    ax.set_xlim(left=0.4)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_forest.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Prevalence + Protection gap ---
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    grp_labels = ['No\nNoise', 'Occup\nOnly', 'Leisure\nOnly', 'Both']
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    # Panel A: Tinnitus sleep disturbance
    prev_a = []
    for grp in [0, 1, 2, 3]:
        sub = s1[s1['exp_group'] == grp]
        t = sub['WTFA_SA'].sum()
        ev = sub.loc[sub['tin_sleep'] == 1, 'WTFA_SA'].sum()
        prev_a.append(ev / t * 100 if t > 0 else 0)

    bars = axes[0].bar(grp_labels, prev_a, color=colors,
                       edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, prev_a):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Weighted Prevalence (%)', fontsize=11)
    axes[0].set_title('A. Tinnitus Sleep Disturbance\n(Tinnitus Subsample)',
                      fontsize=11)
    axes[0].set_ylim(0, max(prev_a) * 1.25)

    # Panel B: Short sleep
    prev_b = []
    for grp in [0, 1, 2, 3]:
        sub = s2[s2['exp_group'] == grp]
        t = sub['WTFA_SA'].sum()
        ev = sub.loc[sub['short_sleep'] == 1, 'WTFA_SA'].sum()
        prev_b.append(ev / t * 100)

    bars2 = axes[1].bar(grp_labels, prev_b, color=colors,
                        edgecolor='black', linewidth=0.5)
    for b, v in zip(bars2, prev_b):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                     f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Weighted Prevalence (%)', fontsize=11)
    axes[1].set_title('B. Short Sleep (<7h)\n(Full Adult Sample)', fontsize=11)
    axes[1].set_ylim(0, max(prev_b) * 1.25)

    # Panel C: Hearing protection gap
    occ_sub = s2[(s2['occ_noise'] == 1) & (s2['work_prot'].notna())]
    occ_prot = (occ_sub.loc[occ_sub['work_prot'] == 1, 'WTFA_SA'].sum() /
                occ_sub['WTFA_SA'].sum() * 100)

    lei_sub = s2[(s2['lei_noise'] == 1) & (s2['HRLSP1'].notna())]
    lei_prot = (lei_sub.loc[recode_prot_for_fig(lei_sub['HRLSP1']) == 1,
                'WTFA_SA'].sum() / lei_sub['WTFA_SA'].sum() * 100)

    x = np.arange(2)
    axes[2].bar(x, [occ_prot, lei_prot], 0.4,
                label='Regular', color='#27ae60',
                edgecolor='black', linewidth=0.5)
    axes[2].bar(x, [100 - occ_prot, 100 - lei_prot], 0.4,
                bottom=[occ_prot, lei_prot],
                label='Irregular/None', color='#e74c3c',
                edgecolor='black', linewidth=0.5)

    for i, (p_val, n_val) in enumerate([
        (occ_prot, 100 - occ_prot),
        (lei_prot, 100 - lei_prot)
    ]):
        axes[2].text(i, p_val / 2, f'{p_val:.0f}%',
                     ha='center', va='center', fontsize=11,
                     fontweight='bold', color='white')
        axes[2].text(i, p_val + n_val / 2, f'{n_val:.0f}%',
                     ha='center', va='center', fontsize=11,
                     fontweight='bold', color='white')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['Occupational\nworkers', 'Leisure\nexposed'])
    axes[2].set_ylabel('Percentage (%)', fontsize=11)
    axes[2].set_title('C. Hearing Protection Use\n(Noise-Exposed Adults)',
                      fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_prevalence.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Figures saved to output/")


# ============================================================
if __name__ == '__main__':
    main()
