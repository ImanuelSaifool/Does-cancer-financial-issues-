import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor # NEW: HistGradientBoosting
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, median_absolute_error

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2014 = pd.read_sas(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h171.ssp", format='xport', encoding='utf-8')
df2014["YEAR"] = 2014

df2015 = pd.read_sas(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h181.ssp", format='xport', encoding='utf-8')
df2015["YEAR"] = 2015

df2016 = pd.read_sas(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h192.ssp", format='xport', encoding='utf-8')
df2016["YEAR"] = 2016

df2017 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h201.csv")
df2017["YEAR"] = 2017

df2018 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h209.csv")
df2018["YEAR"] = 2018

df2019 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h216.csv")
df2019["YEAR"] = 2019

df2020 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\H224.csv")
df2020["YEAR"] = 2020

df2021 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h233.csv")
df2021["YEAR"] = 2021

df2022 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h243.csv")
df2022["YEAR"] = 2022

df2023 = pd.read_csv(r"C:\Users\imanu\OneDrive\Desktop\Coding Projects\h251.csv")
df2023["YEAR"] = 2023

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. STANDARDIZING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Treating inflation
df2014["FAMINC14"] = df2014["FAMINC14"] * 1.30
df2014["TOTSLF14"] = df2014["TOTSLF14"] * 1.30

df2015["FAMINC15"] = df2015["FAMINC15"] * 1.28
df2015["TOTSLF15"] = df2015["TOTSLF15"] * 1.28

df2016["FAMINC16"] = df2016["FAMINC16"] * 1.26
df2016["TOTSLF16"] = df2016["TOTSLF16"] * 1.26

df2017["FAMINC17"] = df2017["FAMINC17"] * 1.25
df2017["TOTSLF17"] = df2017["TOTSLF17"] * 1.25

df2018["FAMINC18"] = df2018["FAMINC18"] * 1.22
df2018["TOTSLF18"] = df2018["TOTSLF18"] * 1.22

df2019["FAMINC19"] = df2019["FAMINC19"] * 1.19
df2019["TOTSLF19"] = df2019["TOTSLF19"] * 1.19

df2020["FAMINC20"] = df2020["FAMINC20"] * 1.17
df2020["TOTSLF20"] = df2020["TOTSLF20"] * 1.17

df2021['FAMINC'] = df2021['FAMINC'] * 1.12
df2021['TOTSLF'] = df2021['TOTSLF'] * 1.12

df2022['FAMINC'] = df2022['FAMINC'] * 1.04
df2022['TOTSLF'] = df2022['TOTSLF'] * 1.04

df2014 = df2014.rename(columns={"TOTSLF14": "TOTSLF", "FAMINC14": "FAMINC", "TOTMCD14": "TOTMCD", "TOTMCR14": "TOTMCR", "TOTVA14": "TOTVA", "TOTTRI14": "TOTTRI", "TOTOFD14": "TOTOFD", "TOTSTL14": "TOTSTL", "REGION14": "REGION", "PRVEV14": "PRVEV", "POVCAT14": "POVCAT", "FOODST14": "FOODST", "DDNWRK14": "DDNWRK", "FAMSZE14": "FAMSZE"})
df2015 = df2015.rename(columns={"TOTSLF15": "TOTSLF", "FAMINC15": "FAMINC", "TOTMCD15": "TOTMCD", "TOTMCR15": "TOTMCR", "TOTVA15": "TOTVA", "TOTTRI15": "TOTTRI", "TOTOFD15": "TOTOFD", "TOTSTL15": "TOTSTL", "REGION15": "REGION", "PRVEV15": "PRVEV", "POVCAT15": "POVCAT", "FOODST15": "FOODST", "DDNWRK15": "DDNWRK", "FAMSZE15": "FAMSZE"})
df2016 = df2016.rename(columns={"TOTSLF16": "TOTSLF", "FAMINC16": "FAMINC", "TOTMCD16": "TOTMCD", "TOTMCR16": "TOTMCR", "TOTVA16": "TOTVA", "TOTTRI16": "TOTTRI", "TOTOFD16": "TOTOFD", "TOTSTL16": "TOTSTL", "REGION16": "REGION", "PRVEV16": "PRVEV", "POVCAT16": "POVCAT", "FOODST16": "FOODST", "DDNWRK16": "DDNWRK", "FAMSZE16": "FAMSZE"})
df2017 = df2017.rename(columns={"TOTSLF17": "TOTSLF", "FAMINC17": "FAMINC", "TOTMCD17": "TOTMCD", "TOTMCR17": "TOTMCR", "TOTVA17": "TOTVA", "TOTTRI17": "TOTTRI", "TOTOFD17": "TOTOFD", "TOTSTL17": "TOTSTL", "REGION17": "REGION", "PRVEV17": "PRVEV", "POVCAT17": "POVCAT", "FOODST17": "FOODST", "DDNWRK17": "DDNWRK", "FAMSZE17": "FAMSZE"})
df2018 = df2018.rename(columns={"TOTSLF18": "TOTSLF", "FAMINC18": "FAMINC", "TOTMCD18": "TOTMCD", "TOTMCR18": "TOTMCR", "TOTVA18": "TOTVA", "TOTTRI18": "TOTTRI", "TOTOFD18": "TOTOFD", "TOTSTL18": "TOTSTL", "REGION18": "REGION", "PRVEV18": "PRVEV", "POVCAT18": "POVCAT", "FOODST18": "FOODST", "DDNWRK18": "DDNWRK", "FAMSZE18": "FAMSZE"})
df2019 = df2019.rename(columns={"TOTSLF19": "TOTSLF", "FAMINC19": "FAMINC", "TOTMCD19": "TOTMCD", "TOTMCR19": "TOTMCR", "TOTVA19": "TOTVA", "TOTTRI19": "TOTTRI", "TOTOFD19": "TOTOFD", "TOTSTL19": "TOTSTL", "REGION19": "REGION", "PRVEV19": "PRVEV", "POVCAT19": "POVCAT", "FOODST19": "FOODST", "DDNWRK19": "DDNWRK", "FAMSZE19": "FAMSZE"})
df2020 = df2020.rename(columns={"TOTSLF20": "TOTSLF", "FAMINC20": "FAMINC", "TOTMCD20": "TOTMCD", "TOTMCR20": "TOTMCR", "TOTVA20": "TOTVA", "TOTTRI20": "TOTTRI", "TOTOFD20": "TOTOFD", "TOTSTL20": "TOTSTL", "REGION20": "REGION", "PRVEV20": "PRVEV", "POVCAT20": "POVCAT", "FOODST20": "FOODST", "DDNWRK20": "DDNWRK", "FAMSZE20": "FAMSZE"})
df2021 = df2021.rename(columns={"TOTSLF21": "TOTSLF", "FAMINC21": "FAMINC", "TOTMCD21": "TOTMCD", "TOTMCR21": "TOTMCR", "TOTVA21": "TOTVA", "TOTTRI21": "TOTTRI", "TOTOFD21": "TOTOFD", "TOTSTL21": "TOTSTL", "REGION21": "REGION", "PRVEV21": "PRVEV", "POVCAT21": "POVCAT", "FOODST21": "FOODST", "DDNWRK21": "DDNWRK", "FAMSZE21": "FAMSZE"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF", "FAMINC22": "FAMINC", "TOTMCD22": "TOTMCD", "TOTMCR22": "TOTMCR", "TOTVA22": "TOTVA", "TOTTRI22": "TOTTRI", "TOTOFD22": "TOTOFD", "TOTSTL22": "TOTSTL", "REGION22": "REGION", "PRVEV22": "PRVEV", "POVCAT22": "POVCAT", "FOODST22": "FOODST", "DDNWRK22": "DDNWRK", "FAMSZE22": "FAMSZE"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF", "FAMINC23": "FAMINC", "TOTMCD23": "TOTMCD", "TOTMCR23": "TOTMCR", "TOTVA23": "TOTVA", "TOTTRI23": "TOTTRI", "TOTOFD23": "TOTOFD", "TOTSTL23": "TOTSTL", "REGION23": "REGION", "PRVEV23": "PRVEV", "POVCAT23": "POVCAT", "FOODST23": "FOODST", "DDNWRK23": "DDNWRK", "FAMSZE23": "FAMSZE"})

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2.5 MEMORY OPTIMIZATION (Pre-Filtering to Prevent Cartesian Explosion)
# ----------------------------------------------------------------------------------------------------------------------------------------------
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX", "REGION"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
insurance_features = ["TOTMCD", "TOTMCR", "TOTVA", "TOTTRI", "TOTOFD", "TOTSTL"]
medicaid = ["TOTMCD"]

adherance_prefixes = ["DLAYCA", "AFRDCA", "DLAYPM", "AFRDPM"]
disease_prefixes = ["DIABDX", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON", "ARTHDX", "ARTHTYPE", "OHRTTYPE", "BPMLDX"]
age_diag_prefixes = ["DIABAGED", "HIBPAGED", "CHDAGED", "ANGIAGED", "MIAGED", "OHRTAGED", "STRKAGED", "CHOLAGED", "EMPHAGED", "ASTHAGED", "ARTHAGED"]
race_prefixes = ["RACETHX", "RACEV1X", "RACEV2X"]
income_prefixes = ["TTLP", "WAGEP"]
sdoh_prefixes = ["FAMSZE", "PRVEV", "RTHLTH", "MNHLTH", "POVCAT", "FOODST", "EMPST", "DDNWRK", "ADLHLP", "PHQ2", "K6SUM", "ADDPRS", "ADNERV", "ADINSB", "ADOVER", "ACTLIM", "WLKLIM"]
util_prefixes = ["IPDIS", "IPNGTD", "ERTOT", "OBTOTV", "OPTOTV", "RXTOT"] # Added IPNGTD (Hospital Nights)

keep_prefixes = tuple(
    demog_features + cancer_features + insurance_features + adherance_prefixes + 
    disease_prefixes + age_diag_prefixes + race_prefixes + income_prefixes + 
    sdoh_prefixes + util_prefixes + ["DUPERSID", "CANCERDX", "CANCEREX", "YEAR"]
)

raw_dfs = [df2014, df2015, df2016, df2017, df2018, df2019, df2020, df2021, df2022, df2023]
filtered_dfs = []

for df in raw_dfs:
    cols_to_keep = [c for c in df.columns if c.startswith(keep_prefixes)]
    filtered_dfs.append(df[cols_to_keep])

main_df = pd.concat(filtered_dfs, axis=0, ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING
# ----------------------------------------------------------------------------------------------------------------------------------------------
new_columns_dict = {}

for prefixes, list_name in zip(
    [adherance_prefixes, disease_prefixes, age_diag_prefixes, race_prefixes, income_prefixes, util_prefixes],
    ['adherance_features', 'other_disease_features', 'age_diag_features', 'race_features', 'income_features', 'util_features']
):
    feature_list = []
    for pref in prefixes:
        cols = [c for c in main_df.columns if c.startswith(pref)]
        if cols:
            new_columns_dict[pref] = main_df[cols].bfill(axis=1).iloc[:, 0]
            feature_list.append(pref)
    globals()[list_name] = feature_list

if new_columns_dict:
    overlap = [c for c in new_columns_dict.keys() if c in main_df.columns]
    if overlap:
        main_df = main_df.drop(columns=overlap)
        
    main_df = pd.concat([main_df, pd.DataFrame(new_columns_dict)], axis=1)

features = demog_features + cancer_features + other_disease_features + adherance_features + insurance_features + age_diag_features + race_features + income_features + util_features

cancer_col = 'CANCERDX' if 'CANCERDX' in main_df.columns else ('CANCEREX' if 'CANCEREX' in main_df.columns else None)

if cancer_col:
    clean_df = main_df[(main_df[cancer_col] == 1) & (main_df['TOTMCD'] > 0)].copy()
else:
    clean_df = main_df[main_df['TOTMCD'] > 0].copy()

clean_df = clean_df.drop_duplicates(subset=['DUPERSID'], keep='first')
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]
clean_df[cancer_features] = clean_df[cancer_features].replace([-1,-7, -8, -9], 2)

for col in income_features:
    clean_df[col] = clean_df[col].replace([-15, -1, -7, -8, -9], np.nan)
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())

for col in age_diag_features:
    clean_df[col] = clean_df[col].replace([-15, -7, -8, -9], np.nan)
    clean_df[col] = clean_df[col].replace(-1, 0)
    if not clean_df[col].isna().all():
        median_age = clean_df[clean_df[col] > 0][col].median()
        clean_df[col] = clean_df[col].fillna(median_age if pd.notnull(median_age) else 0)

for col in race_features:
    clean_df[col] = clean_df[col].replace([-15, -1, -7, -8, -9], np.nan)
    clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])

for col in util_features:
    clean_df[col] = clean_df[col].replace([-15, -1, -7, -8, -9], np.nan)
    clean_df[col] = clean_df[col].fillna(0)

def clean_adherence(val):
    if val == 1: return 1
    elif val == 2: return 0
    else: return np.nan

for col in adherance_features:
    clean_df[col] = clean_df[col].apply(clean_adherence)

clean_df = clean_df.dropna(subset=adherance_features)
clean_df['TOXICITY_SCORE'] = clean_df[adherance_features].sum(axis=1)

def calculate_toxicity_tier(row):
    if ('AFRDCA' in row and row['AFRDCA'] == 1) or ('AFRDPM' in row and row['AFRDPM'] == 1): return "Severe (Forgone Care/Meds)"
    elif ('DLAYCA' in row and row['DLAYCA'] == 1) or ('DLAYPM' in row and row['DLAYPM'] == 1): return "Moderate (Delayed Care/Meds)"
    else: return "None (Fully Adherent)"

clean_df['TOXICITY_TIER'] = clean_df.apply(calculate_toxicity_tier, axis=1)

clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)
clean_df['MCD_TOTAL'] = clean_df[medicaid].sum(axis=1)

clean_df['TOTAL_KNOWN_COST'] = clean_df['PUBLIC_TOTAL'] + clean_df['TOTSLF']
clean_df['COVERAGE_RATIO'] = clean_df['MCD_TOTAL'] / (clean_df['TOTAL_KNOWN_COST'] + 1e-9)
clean_df['COVERAGE_RATIO_PCT'] = clean_df['COVERAGE_RATIO'] * 100
clean_df['CATASTROPHIC_COST'] = (clean_df['TOTSLF'] > (0.10 * clean_df['FAMINC'])).astype(int)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.5 FEATURE ENGINEERING (Insurance & Geography Proxies)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['IS_MEDICARE_AGE'] = (clean_df['AGELAST'] >= 65).astype(int)
clean_df['IS_CHIP_AGE'] = (clean_df['AGELAST'] <= 19).astype(int)
clean_df['IS_VETERAN'] = (clean_df['TOTVA'] > 0).astype(int)
clean_df['IS_MILITARY_FAM'] = (clean_df['TOTTRI'] > 0).astype(int)
clean_df['IS_FED_WORKER'] = (clean_df['TOTOFD'] > 0).astype(int)

clean_df['REGION_NORTHEAST'] = (clean_df['REGION'] == 1).astype(int)
clean_df['REGION_MIDWEST'] = (clean_df['REGION'] == 2).astype(int)
clean_df['REGION_SOUTH'] = (clean_df['REGION'] == 3).astype(int)
clean_df['REGION_WEST'] = (clean_df['REGION'] == 4).astype(int)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.6 FEATURE ENGINEERING (Safe Socioeconomic & Health Proxies)
# ----------------------------------------------------------------------------------------------------------------------------------------------
print("\n[*] Scanning dataset for MEPS socioeconomic and depression variables...")

target_extra_cols = {
    'FAMSZE': 'FAMILY_SIZE',
    'PRVEV': 'HAS_PRIVATE_INS',
    'RTHLTH': 'PERCEIVED_PHYS_HLTH',
    'MNHLTH': 'PERCEIVED_MENTAL_HLTH',
    'POVCAT': 'POVERTY_CATEGORY',
    'FOODST': 'FOOD_STAMPS',
    'EMPST': 'EMPLOYMENT_STATUS',
    'DDNWRK': 'DAYS_MISSED_WORK',
    'ADLHLP': 'ADL_HELP_NEEDED',
    'PHQ2': 'PHQ2_DEPRESSION_SCORE',
    'K6SUM': 'KESSLER_DISTRESS_INDEX',
    'ADDPRS': 'FREQ_DEPRESSED',
    'ADNERV': 'FREQ_NERVOUS',
    'ADINSB': 'BELIEF_INS_NOT_WORTH_COST',
    'ADOVER': 'BELIEF_CAN_OVERCOME_ILLNESS',
    'ACTLIM': 'WORK_HOUSEWORK_LIMITATION',
    'WLKLIM': 'PHYSICAL_FUNCTIONING_LIMITATION'
}

available_extras = []
for original_col, new_name in target_extra_cols.items():
    matching_cols = [c for c in clean_df.columns if original_col in c]
    if matching_cols:
        clean_df[new_name] = clean_df[matching_cols].bfill(axis=1).iloc[:, 0]
        clean_df[new_name] = clean_df[new_name].replace([-1, -7, -8, -9], np.nan)
        if not clean_df[new_name].isna().all():
            clean_df[new_name] = clean_df[new_name].fillna(clean_df[new_name].median())
            available_extras.append(new_name)
            print(f"    - Found and cleaned: {new_name}")
        else:
            clean_df = clean_df.drop(columns=[new_name])

if 'FAMILY_SIZE' in available_extras:
    clean_df['INCOME_PER_CAPITA'] = clean_df['FAMINC'] / clean_df['FAMILY_SIZE'].replace(0, 1)
    available_extras.append('INCOME_PER_CAPITA')
    print("    - Engineered: INCOME_PER_CAPITA")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.7 FEATURE ENGINEERING (Healthcare Utilization Intensity)
# ------------------------------------------------------------------------------

print("\n[*] Engineering healthcare utilization intensity variables...")

# Total encounters across settings
clean_df['TOTAL_VISITS'] = (
    clean_df['ERTOT'] +
    clean_df['OBTOTV'] +
    clean_df['OPTOTV']
)

# Total inpatient burden
clean_df['INPATIENT_BURDEN'] = (
    clean_df['IPDIS'] +
    clean_df['IPNGTD']
)

# Medication intensity
clean_df['RX_INTENSITY'] = clean_df['RXTOT']

# Encounters per year proxy
clean_df['CARE_INTENSITY_INDEX'] = (
    clean_df['TOTAL_VISITS'] +
    clean_df['INPATIENT_BURDEN'] +
    clean_df['RX_INTENSITY']
)

# Emergency dependency
clean_df['ER_DEPENDENCY'] = clean_df['ERTOT'] / (clean_df['TOTAL_VISITS'] + 1e-6)

# Outpatient vs inpatient balance
clean_df['OUTPATIENT_RATIO'] = (
    clean_df['OBTOTV'] + clean_df['OPTOTV']
) / (clean_df['CARE_INTENSITY_INDEX'] + 1e-6)

# Chronic medication reliance
clean_df['RX_PER_VISIT'] = clean_df['RXTOT'] / (clean_df['TOTAL_VISITS'] + 1e-6)

util_engineered_features = [
    'TOTAL_VISITS',
    'INPATIENT_BURDEN',
    'RX_INTENSITY',
    'CARE_INTENSITY_INDEX',
    'ER_DEPENDENCY',
    'OUTPATIENT_RATIO',
    'RX_PER_VISIT'
]

print(f"    - Added {len(util_engineered_features)} utilization intensity features.")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 14.8 FEATURE ENGINEERING (Year Effects)
# ------------------------------------------------------------------------------

print("\n[*] Creating year fixed effects...")

year_dummies = pd.get_dummies(clean_df['YEAR'], prefix="YEAR")

clean_df = pd.concat([clean_df, year_dummies], axis=1)

year_features = year_dummies.columns.tolist()

print(f"   - Added {len(year_features)} year effect variables.")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 15. PRESCRIPTIVE MODELING (Automated Feature Selection + RF Regressor)
# ----------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "="*80)
print("INITIALIZING PROACTIVE SUBSIDY CALCULATOR (FEATURE SELECTION + POISSON-GBM)")
print("="*80)

success_df = clean_df[clean_df['TOXICITY_TIER'] == "None (Fully Adherent)"].copy()
regional_features = ['REGION_NORTHEAST', 'REGION_MIDWEST', 'REGION_SOUTH', 'REGION_WEST']

ml_features = [
    'FAMINC', 'TOTSLF', 'CATASTROPHIC_COST', 'AGELAST', 'SEX', 
    'IS_MEDICARE_AGE', 'IS_CHIP_AGE', 'IS_VETERAN', 'IS_MILITARY_FAM', 'IS_FED_WORKER'
] + regional_features + cancer_features + other_disease_features + age_diag_features + race_features + income_features + util_features + util_engineered_features + available_extras + year_features

ml_df = success_df.dropna(subset=ml_features + ['PUBLIC_TOTAL']).copy()

X = ml_df[ml_features]
y_true = ml_df['PUBLIC_TOTAL']

cap_value = y_true.quantile(0.90)
print(f"[*] Capping extreme catastrophic outliers at the 90th percentile: ${cap_value:,.2f}")

# Keep the cap, but DO NOT log-transform the final target
y_clipped = np.clip(y_true, a_min=0, a_max=cap_value)
y_log_for_scout = np.log1p(y_clipped) # Keep log ONLY for the scout model to prevent outlier-bias during feature selection

# Split using the true dollar values and the log values
X_train, X_test, y_train_dollars, y_test_dollars, y_train_log, y_test_log = train_test_split(
    X, y_clipped, y_log_for_scout, test_size=0.2, random_state=42
)

print("\n[*] Phase 1: Training Scout Model to identify and drop noisy features...")
# Train scout on log values so feature selection isn't hijacked by extreme dollar outliers
scout_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
scout_rf.fit(X_train, y_train_log)

# Use SelectFromModel to strip away features that have an importance below the median
selector = SelectFromModel(scout_rf, prefit=True, threshold='median')
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)
selected_features_list = X.columns[selected_indices].tolist()

print(f"    - Reduced features from {X.shape[1]} down to the top {len(selected_features_list)} highly predictive signals.")

importances = scout_rf.feature_importances_
top_5_idx = np.argsort(importances)[-5:][::-1]
print(f"    - Top 5 strongest predictors: {[X.columns[i] for i in top_5_idx]}")

print("\n[*] Phase 2: Training Optimized Gradient Booster with Poisson Loss...")
# Swap to Poisson loss for the final model and train on TRUE DOLLARS
final_rf_model = HistGradientBoostingRegressor(
    loss='poisson', # <--- THIS FIXES THE UNDERESTIMATION
    max_depth=10,
    learning_rate=0.05,
    max_iter=500,
    min_samples_leaf=20
)

# Fit on un-logged dollars
final_rf_model.fit(X_train_selected, y_train_dollars)

# Predict directly in dollars (no np.expm1 needed!)
y_pred_dollars = final_rf_model.predict(X_test_selected)

mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
mse = mean_squared_error(y_test_dollars, y_pred_dollars)
rmse = mse ** 0.5
nmae = mae / y_test_dollars.mean()

print(f"\n--- Model Ready (Poisson HistGradientBoostingRegressor) ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f} (Average variation, penalizing large errors)")
print(f"Normalized MAE: {nmae:.2f}")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# MODEL EVALUATION METRICS
# ------------------------------------------------------------------------------

r2 = r2_score(y_test_dollars, y_pred_dollars)
medae = median_absolute_error(y_test_dollars, y_pred_dollars)

# Mean Absolute Percentage Error (avoid divide by zero)
mape = np.mean(np.abs((y_test_dollars - y_pred_dollars) / (y_test_dollars + 1e-9))) * 100

print("\n--- Additional Evaluation Metrics ---")
print(f"R² Score (Explained Variance): {r2:.3f}")
print(f"Median Absolute Error: ${medae:,.2f}")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# RESIDUAL ANALYSIS
# ------------------------------------------------------------------------------

residuals = y_test_dollars - y_pred_dollars

print("\n[*] Generating diagnostic plots...")

# 1. Prediction vs Actual
plt.figure(figsize=(7,6))
plt.scatter(y_test_dollars, y_pred_dollars, alpha=0.4)
plt.plot([y_test_dollars.min(), y_test_dollars.max()], [y_test_dollars.min(), y_test_dollars.max()])
plt.xlabel("Actual Medicaid Spending")
plt.ylabel("Predicted Medicaid Spending")
plt.title("Predicted vs Actual Medicaid Spending")
plt.show()

# 2. Residual distribution
plt.figure(figsize=(7,6))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error ($)")
plt.ylabel("Frequency")
plt.show()

# 3. Residuals vs Predictions
plt.figure(figsize=(7,6))
plt.scatter(y_pred_dollars, residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("Predicted Spending")
plt.ylabel("Residuals (Error)")
plt.title("Residuals vs Predicted Spending")
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# DECILE CALIBRATION PLOT
# ------------------------------------------------------------------------------
print("\n[*] Generating Decile Calibration Plot...")

# Create a temporary dataframe for analysis
decile_df = pd.DataFrame({
    'Actual': y_test_dollars,
    'Predicted': y_pred_dollars
})

# Divide the data into 10 deciles based on Predicted Values
# using rank(method='first') to safely handle many identical predictions (e.g., ties at exactly 0 or median limits)
decile_df['Decile'] = pd.qcut(decile_df['Predicted'].rank(method='first'), 10, labels=False) + 1

# Calculate the mean Actual and mean Predicted for each decile
decile_stats = decile_df.groupby('Decile')[['Actual', 'Predicted']].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
bar_width = 0.35
r1 = np.arange(len(decile_stats['Decile']))
r2 = [x + bar_width for x in r1]

plt.bar(r1, decile_stats['Actual'], color='#2E86AB', width=bar_width, edgecolor='white', label='Actual Average Spending')
plt.bar(r2, decile_stats['Predicted'], color='#F18F01', width=bar_width, edgecolor='white', label='Predicted Average Spending')

plt.xlabel('Patient Risk Decile (1 = Lowest Predicted Cost, 10 = Highest Predicted Cost)', fontweight='bold')
plt.ylabel('Average Spending ($)', fontweight='bold')
plt.title('Model Calibration: Actual vs. Predicted Spending by Decile')
plt.xticks([r + bar_width/2 for r in range(len(decile_stats['Decile']))], decile_stats['Decile'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 16. INTERACTIVE CLINICAL DECISION SUPPORT TOOL
# ----------------------------------------------------------------------------------------------------------------------------------------------
def run_subsidy_calculator():
    print("\n" + "="*80)
    print(" CLINICAL DECISION SUPPORT: FULL-YEAR EXPLANATORY SUBSIDY CALCULATOR")
    print("="*80)
    print("Type 'quit' at any prompt to exit the tool.\n")

    cancer_list = ["Bladder", "Breast", "Cervix", "Colon", "Lung", "Lymphoma", "Melanoma", "Other", "Prostate", "Skin (Non-Melanoma)", "Skin (Unknown)", "Uterus"]
    disease_list = ["Diabetes", "High Blood Pressure", "Coronary Heart Disease", "Angina", "Heart Attack", "Other Heart Disease", "Stroke", "High Cholesterol", "Emphysema", "Asthma", "Chronic Bronchitis", "Arthritis", "Arthritis Subtype", "Heart Subtype", "Multiple BP"]
    region_names = ["Northeast", "Midwest", "South", "West"]

    while True:
        try:
            faminc_in = input("Enter Patient's Current Family Income ($): ").strip().replace(',', '').replace('$', '')
            if faminc_in.lower() == 'quit': break
            patient_faminc = float(faminc_in)

            totslf_in = input("Enter Final Annual Out-of-Pocket Cost for Treatment ($): ").strip().replace(',', '').replace('$', '')
            if totslf_in.lower() == 'quit': break
            patient_totslf = float(totslf_in)
            
            catastrophic_cost = 1 if patient_totslf > (0.10 * patient_faminc) else 0

            age_in = input("Enter Patient's Current Age: ").strip()
            if age_in.lower() == 'quit': break
            patient_age = int(age_in)

            sex_in = input("Enter Assigned Sex (1 = Male, 2 = Female): ").strip()
            if sex_in.lower() == 'quit': break
            patient_sex = int(sex_in)

            print("\n--- SOCIAL DETERMINANTS OF HEALTH (SDoH) ---")
            
            patient_famsze = 1
            if 'FAMILY_SIZE' in available_extras:
                fs_in = input("Enter Family Size (number of people in household): ").strip()
                if fs_in.lower() == 'quit': break
                patient_famsze = int(fs_in) if fs_in.isdigit() else 1

            patient_prv = 2
            if 'HAS_PRIVATE_INS' in available_extras:
                prv_in = input("Does the patient have any Private Insurance? (y/n): ").strip().lower()
                if prv_in == 'quit': break
                patient_prv = 1 if prv_in == 'y' else 2
                
            patient_pov = 3
            if 'POVERTY_CATEGORY' in available_extras:
                pov_in = input("Enter Poverty Category (1=Poor to 5=High Income): ").strip()
                if pov_in.lower() == 'quit': break
                patient_pov = int(pov_in) if pov_in.isdigit() else 3

            patient_foodst = 2
            if 'FOOD_STAMPS' in available_extras:
                fs_in = input("Does the patient receive Food Stamps/SNAP? (y/n): ").strip().lower()
                if fs_in == 'quit': break
                patient_foodst = 1 if fs_in == 'y' else 2
                
            patient_ddnwrk = 0
            if 'DAYS_MISSED_WORK' in available_extras:
                dw_in = input("Estimated days of work missed due to illness this year: ").strip()
                if dw_in.lower() == 'quit': break
                patient_ddnwrk = int(dw_in) if dw_in.isdigit() else 0

            patient_adl = 2
            if 'ADL_HELP_NEEDED' in available_extras:
                adl_in = input("Does the patient need help with daily activities (bathing, etc)? (y/n): ").strip().lower()
                if adl_in == 'quit': break
                patient_adl = 1 if adl_in == 'y' else 2
                
            patient_phq2 = 0
            if 'PHQ2_DEPRESSION_SCORE' in available_extras:
                phq_in = input("PHQ-2 Depression Score (0 to 6): ").strip()
                if phq_in.lower() == 'quit': break
                patient_phq2 = int(phq_in) if phq_in.isdigit() else 0

            patient_ph = clean_df['PERCEIVED_PHYS_HLTH'].median() if 'PERCEIVED_PHYS_HLTH' in available_extras else 3
            patient_mh = clean_df['PERCEIVED_MENTAL_HLTH'].median() if 'PERCEIVED_MENTAL_HLTH' in available_extras else 3 
            patient_empst = clean_df['EMPLOYMENT_STATUS'].median() if 'EMPLOYMENT_STATUS' in available_extras else 1
            
            patient_race = {}
            for col in race_features: patient_race[col] = 1 
                
            patient_income = {}
            for col in income_features: patient_income[col] = clean_df[col].median() if col in clean_df.columns else 0 
            
            print("\n--- HEALTHCARE UTILIZATION (PAST 12 MONTHS) ---")
            ip_in = input("Number of Inpatient Hospital Discharges: ").strip()
            if ip_in.lower() == 'quit': break
            patient_ipdis = int(ip_in) if ip_in.isdigit() else 0
            
            ipn_in = input("Total Nights Spent in Hospital: ").strip()
            if ipn_in.lower() == 'quit': break
            patient_ipngtd = int(ipn_in) if ipn_in.isdigit() else 0
            
            er_in = input("Number of Emergency Room Visits: ").strip()
            if er_in.lower() == 'quit': break
            patient_ertot = int(er_in) if er_in.isdigit() else 0

            patient_util = {
                'IPDIS': [patient_ipdis],
                'IPNGTD': [patient_ipngtd],
                'ERTOT': [patient_ertot],
                'OBTOTV': [clean_df['OBTOTV'].median() if 'OBTOTV' in clean_df.columns else 0],
                'OPTOTV': [clean_df['OPTOTV'].median() if 'OPTOTV' in clean_df.columns else 0],
                'RXTOT': [clean_df['RXTOT'].median() if 'RXTOT' in clean_df.columns else 0]
            }

            print("\n--- GEOGRAPHY ---")
            for i, r in enumerate(region_names): print(f"{i+1}. {r}")
            region_choice = input("Select Patient's US Region (1-4): ").strip()
            if region_choice.lower() == 'quit': break
            region_idx = int(region_choice)
            
            patient_region = {
                'REGION_NORTHEAST': [1 if region_idx == 1 else 0],
                'REGION_MIDWEST': [1 if region_idx == 2 else 0],
                'REGION_SOUTH': [1 if region_idx == 3 else 0],
                'REGION_WEST': [1 if region_idx == 4 else 0]
            }

            vet_in = input("Is the patient a US Veteran? (y/n): ").strip().lower()
            if vet_in == 'quit': break
            patient_vet = 1 if vet_in == 'y' else 0

            mil_in = input("Is the patient/family in the military [Tricare eligible]? (y/n): ").strip().lower()
            if mil_in == 'quit': break
            patient_mil = 1 if mil_in == 'y' else 0

            fed_in = input("Does the patient work for the Federal Government? (y/n): ").strip().lower()
            if fed_in == 'quit': break
            patient_fed = 1 if fed_in == 'y' else 0

            patient_medicare = 1 if patient_age >= 65 else 0
            patient_chip = 1 if patient_age <= 19 else 0

            print("\n--- PRIMARY CANCER DIAGNOSIS ---")
            for i, c in enumerate(cancer_list): print(f"{i+1}. {c}")
            cancer_choice = input("Select Primary Cancer Type (1-12): ").strip()
            if cancer_choice.lower() == 'quit': break
            
            patient_cancers = {col: 2 for col in cancer_features}
            if 1 <= int(cancer_choice) <= 12:
                selected_cancer_col = cancer_features[int(cancer_choice) - 1]
                patient_cancers[selected_cancer_col] = 1

            print("\n--- COMORBIDITIES & AGE OF DIAGNOSIS ---")
            for i, d in enumerate(disease_list): print(f"{i+1}. {d}")
            disease_choice = input("Enter Comorbidities by number (comma separated, e.g., '1, 2, 8') or '0' for None: ").strip()
            if disease_choice.lower() == 'quit': break
            
            patient_diseases = {col: 2 for col in other_disease_features}
            patient_age_diag = {col: 0 for col in age_diag_features}
            
            if disease_choice != '0':
                choices = [int(x.strip()) for x in disease_choice.split(',') if x.strip().isdigit()]
                for choice in choices:
                    if 1 <= choice <= len(other_disease_features):
                        selected_disease_col = other_disease_features[choice - 1]
                        patient_diseases[selected_disease_col] = 1
                        
                        diag_age_prefix = selected_disease_col.replace("DX", "AGED")
                        if diag_age_prefix in patient_age_diag:
                            patient_age_diag[diag_age_prefix] = patient_age

            patient_data = {
                'FAMINC': [patient_faminc], 'TOTSLF': [patient_totslf], 'CATASTROPHIC_COST': [catastrophic_cost],
                'AGELAST': [patient_age], 'SEX': [patient_sex], 'IS_MEDICARE_AGE': [patient_medicare],
                'IS_CHIP_AGE': [patient_chip], 'IS_VETERAN': [patient_vet], 'IS_MILITARY_FAM': [patient_mil],
                'IS_FED_WORKER': [patient_fed]
            }
            
            if 'FAMILY_SIZE' in available_extras:
                patient_data['FAMILY_SIZE'] = [patient_famsze]
                patient_data['INCOME_PER_CAPITA'] = [patient_faminc / max(1, patient_famsze)]
            if 'HAS_PRIVATE_INS' in available_extras: patient_data['HAS_PRIVATE_INS'] = [patient_prv]
            if 'PERCEIVED_PHYS_HLTH' in available_extras: patient_data['PERCEIVED_PHYS_HLTH'] = [patient_ph]
            if 'PERCEIVED_MENTAL_HLTH' in available_extras: patient_data['PERCEIVED_MENTAL_HLTH'] = [patient_mh]
            if 'POVERTY_CATEGORY' in available_extras: patient_data['POVERTY_CATEGORY'] = [patient_pov]
            if 'FOOD_STAMPS' in available_extras: patient_data['FOOD_STAMPS'] = [patient_foodst]
            if 'DAYS_MISSED_WORK' in available_extras: patient_data['DAYS_MISSED_WORK'] = [patient_ddnwrk]
            if 'ADL_HELP_NEEDED' in available_extras: patient_data['ADL_HELP_NEEDED'] = [patient_adl]
            if 'EMPLOYMENT_STATUS' in available_extras: patient_data['EMPLOYMENT_STATUS'] = [patient_empst]
            if 'PHQ2_DEPRESSION_SCORE' in available_extras: patient_data['PHQ2_DEPRESSION_SCORE'] = [patient_phq2]
            
            if 'KESSLER_DISTRESS_INDEX' in available_extras: patient_data['KESSLER_DISTRESS_INDEX'] = [clean_df['KESSLER_DISTRESS_INDEX'].median()]
            if 'FREQ_DEPRESSED' in available_extras: patient_data['FREQ_DEPRESSED'] = [clean_df['FREQ_DEPRESSED'].median()]
            if 'FREQ_NERVOUS' in available_extras: patient_data['FREQ_NERVOUS'] = [clean_df['FREQ_NERVOUS'].median()]
            if 'BELIEF_INS_NOT_WORTH_COST' in available_extras: patient_data['BELIEF_INS_NOT_WORTH_COST'] = [clean_df['BELIEF_INS_NOT_WORTH_COST'].median()]
            if 'BELIEF_CAN_OVERCOME_ILLNESS' in available_extras: patient_data['BELIEF_CAN_OVERCOME_ILLNESS'] = [clean_df['BELIEF_CAN_OVERCOME_ILLNESS'].median()]
            if 'WORK_HOUSEWORK_LIMITATION' in available_extras: patient_data['WORK_HOUSEWORK_LIMITATION'] = [clean_df['WORK_HOUSEWORK_LIMITATION'].median()]
            if 'PHYSICAL_FUNCTIONING_LIMITATION' in available_extras: patient_data['PHYSICAL_FUNCTIONING_LIMITATION'] = [clean_df['PHYSICAL_FUNCTIONING_LIMITATION'].median()]

            patient_data.update(patient_region)
            patient_data.update({k: [v] for k, v in patient_cancers.items()})
            patient_data.update({k: [v] for k, v in patient_diseases.items()})
            patient_data.update({k: [v] for k, v in patient_age_diag.items()})
            patient_data.update({k: [v] for k, v in patient_race.items()})
            patient_data.update({k: [v] for k, v in patient_income.items()})
            
            # --- ENGINEERING INTERACTIVE UTILIZATION PROXIES ---
            total_visits_calc = patient_util['ERTOT'][0] + patient_util['OBTOTV'][0] + patient_util['OPTOTV'][0]
            inpatient_burden_calc = patient_util['IPDIS'][0] + patient_util['IPNGTD'][0]
            rx_intensity_calc = patient_util['RXTOT'][0]
            care_intensity_calc = total_visits_calc + inpatient_burden_calc + rx_intensity_calc
            er_dependency_calc = patient_util['ERTOT'][0] / (total_visits_calc + 1e-6)
            outpatient_ratio_calc = (patient_util['OBTOTV'][0] + patient_util['OPTOTV'][0]) / (care_intensity_calc + 1e-6)
            rx_per_visit_calc = patient_util['RXTOT'][0] / (total_visits_calc + 1e-6)
            
            engineered_util_dict = {
                'TOTAL_VISITS': [total_visits_calc],
                'INPATIENT_BURDEN': [inpatient_burden_calc],
                'RX_INTENSITY': [rx_intensity_calc],
                'CARE_INTENSITY_INDEX': [care_intensity_calc],
                'ER_DEPENDENCY': [er_dependency_calc],
                'OUTPATIENT_RATIO': [outpatient_ratio_calc],
                'RX_PER_VISIT': [rx_per_visit_calc]
            }

            # Year features (Set to median or 0 for simplicity in single-patient inference)
            year_dict = {col: [0] for col in year_features}
            
            patient_data.update(patient_util)
            patient_data.update(engineered_util_dict)
            patient_data.update(year_dict)

            # Create dataframe and select features
            new_patient_df = pd.DataFrame(patient_data)[ml_features]
            new_patient_df_selected = new_patient_df[selected_features_list]

            # --- DIRECT DOLLAR PREDICTION ---
            # The model now predicts directly in dollars, no exponential conversion needed
            recommended_subsidy = final_rf_model.predict(new_patient_df_selected)[0]
            recommended_subsidy = max(0, recommended_subsidy)

            cancer_name = cancer_list[int(cancer_choice) - 1] if 1 <= int(cancer_choice) <= 12 else "Unknown"
            region_name = region_names[region_idx - 1] if 1 <= region_idx <= 4 else "Unknown"
            
            print("\n" + "-" * 80)
            print(" EXPLANATORY PATIENT PROFILE:")
            print(f" Demographics: Age {patient_age} | Sex: {'Male' if patient_sex == 1 else 'Female'} | Region: {region_name}")
            print(f" Clinical: {cancer_name} Cancer | Comorbidities Logged: {'None' if disease_choice == '0' else disease_choice}")
            print(f" Financial: Income ${patient_faminc:,.2f} | Out-of-Pocket: ${patient_totslf:,.2f} | Catastrophic Risk: {'YES' if catastrophic_cost else 'NO'}")
            print(f" Overlapping Coverage: Vet({vet_in.upper()}) | Mil({mil_in.upper()}) | Fed({fed_in.upper()}) | Medicare({'Y' if patient_medicare else 'N'}) | CHIP/State({'Y' if patient_chip else 'N'})")
            if 'PHQ2_DEPRESSION_SCORE' in available_extras:
                print(f" SDoH Flags: PHQ-2 Score [{patient_phq2}] | Missed Work [{patient_ddnwrk} Days] | ADL Help [{'YES' if patient_adl == 1 else 'NO'}]")
            print(f" Utilization: Hospital Stays [{patient_ipdis}] | Hosp Nights [{patient_ipngtd}] | ER Visits [{patient_ertot}]")
            print("-" * 80)
            print(f">>> COMPUTED STATISTICAL SUBSIDY EXPECTATION: ${recommended_subsidy:,.2f} <<<")
            print("    (Estimated public burden based on full-year patient realities)")
            print("-" * 80 + "\n")
            
            run_again = input("Calculate for another patient? (y/n): ").strip().lower()
            if run_again != 'y': break
            print("\n" + "="*80 + "\n")

        except ValueError:
            print("\n[ERROR] Invalid input. Please enter numbers appropriately.\n")
            print("="*80 + "\n")

if __name__ == "__main__":
    run_subsidy_calculator()