import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2019 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/h216.csv")
df2020 = pd.read_csv("https://github.com/ImanuelSaifool/Data-Science-PROJECT-LAB-/raw/main/OneDrive/Desktop/Coding%20Projects/H224.csv")
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. STANDARDIZING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Treating inflation
    # Family income
df2019["FAMINC19"] = df2019["FAMINC19"] * 1.19
df2020["FAMINC20"] = df2020["FAMINC20"] * 1.17
df2021p1['FAMINC21'] = df2021p1['FAMINC21'] * 1.12
df2021p2['FAMINC21'] = df2021p2['FAMINC21'] * 1.12
df2022['FAMINC22'] = df2022['FAMINC22'] * 1.04

df2019["TOTSLF19"] = df2019["TOTSLF19"] * 1.19
df2020["TOTSLF20"] = df2020["TOTSLF20"] * 1.17
df2021p1['TOTSLF21'] = df2021p1['TOTSLF21'] * 1.12
df2021p2['TOTSLF21'] = df2021p2['TOTSLF21'] * 1.12
df2022['TOTSLF22'] = df2022['TOTSLF22'] * 1.04
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Out of pocket cost
df2019 = df2019.rename(columns={"TOTSLF19": "TOTSLF"})
df2020 = df2020.rename(columns={"TOTSLF20": "TOTSLF"})
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# Family income
df2019 = df2019.rename(columns={"FAMINC19": "FAMINC"})
df2020 = df2020.rename(columns={"FAMINC20": "FAMINC"})
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"})
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# Renaming Medicaid
df2019 = df2019.rename(columns={"TOTMCD19": "TOTMCD"})
df2020 = df2020.rename(columns={"TOTMCD20": "TOTMCD"})
df2021p1 = df2021p1.rename(columns={"TOTMCD21": "TOTMCD"})
df2021p2 = df2021p2.rename(columns={"TOTMCD21": "TOTMCD"})
df2022 = df2022.rename(columns={"TOTMCD22": "TOTMCD"})
df2023 = df2023.rename(columns={"TOTMCD23": "TOTMCD"})

# Renaming Region
df2019 = df2019.rename(columns={"REGION19": "REGION"})
df2020 = df2020.rename(columns={"REGION20": "REGION"})
df2021p1 = df2021p1.rename(columns={"REGION21": "REGION"})
df2021p2 = df2021p2.rename(columns={"REGION21": "REGION"})
df2022 = df2022.rename(columns={"REGION22": "REGION"})
df2023 = df2023.rename(columns={"REGION23": "REGION"})
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Combining datasets
main_df = pd.concat([df2019, df2020, df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Define feature lists
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX", "REGION"]
adherance_features = ["DLAYCA42", "AFRDCA42", "DLAYPM42", "AFRDPM42"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER", "CAPROSTA", "CASKINNM", "CASKINDK", "CAUTERUS"]
other_disease_features = ["DIABDX_M18", "HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX", "CHOLDX", "EMPHDX", "ASTHDX", "CHBRON31", "ARTHDX"]
insurance_features = ["TOTMCD"]
Financial_Subjectivity_features = ["PROBPY42", "PYUNBL42", "CRFMPY42"]
features = demog_features + cancer_features + other_disease_features + adherance_features + insurance_features + Financial_Subjectivity_features
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Filters ONLY for patients who actually received Medicaid funding
clean_df = main_df[(main_df['CANCERDX'] == 1) & (main_df['TOTMCD'] > 0)].copy()

# Dropping duplicates for same person
clean_df = clean_df.drop_duplicates(subset=['DUPERSID'], keep='first')

# Filter negative values for demographics to prevent logic error
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

# Filter negative values for cancer features to prevent logic error
clean_df[cancer_features] = clean_df[cancer_features].replace([-1,-7, -8, -9], 2)

for col in Financial_Subjectivity_features:
    if col in clean_df.columns:
        clean_df[col] = clean_df[col].replace([-1, -7, -8, -9], np.nan)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['PUBLIC_TOTAL'] = clean_df[insurance_features].sum(axis=1)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Standardize the adherence features (1 = Issue, 0 = No Issue, NaN = Missing)
def clean_adherence(val):
    if val == 1: 
        return 1  # Yes, experienced financial barrier
    elif val == 2: 
        return 0  # No, did not experience barrier
    else: 
        return np.nan # Treat negatives as missing data

# Apply cleaning to all four features
for col in adherance_features:
    clean_df[col] = clean_df[col].apply(clean_adherence)

# Drop rows where we don't have valid adherence data to avoid skewed math
clean_df = clean_df.dropna(subset=adherance_features)
# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['TOXICITY_SCORE'] = clean_df[adherance_features].sum(axis=1)
def calculate_toxicity_tier(row):
    # If they are completely UNABLE to afford either care or meds -> Severe
    if row['AFRDCA42'] == 1 or row['AFRDPM42'] == 1:
        return "Severe (Forgone Care/Meds)"
    # Else, if they DELAYED care or meds -> Moderate
    elif row['DLAYCA42'] == 1 or row['DLAYPM42'] == 1:
        return "Moderate (Delayed Care/Meds)"
    # Otherwise -> None
    else:
        return "None (Fully Adherent)"

clean_df['TOXICITY_TIER'] = clean_df.apply(calculate_toxicity_tier, axis=1)
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Total Known Cost (What the patient paid + What public insurance paid)
clean_df['TOTAL_KNOWN_COST'] = clean_df['PUBLIC_TOTAL'] + clean_df['TOTSLF']

# 2. Calculate the Coverage Ratio
clean_df['COVERAGE_RATIO'] = clean_df['PUBLIC_TOTAL'] / (clean_df['TOTAL_KNOWN_COST'] + 1e-9)
clean_df['COVERAGE_RATIO_PCT'] = clean_df['COVERAGE_RATIO'] * 100

# ----------------------------------------------------------------------------------------------------------------------------------------------
clean_df['CATASTROPHIC_COST'] = (clean_df['TOTSLF'] > (0.10 * clean_df['FAMINC'])).astype(int)
# ----------------------------------------------------------------------------------------------------------------------------------------------
cancer_map = {
    "CABLADDR": "Bladder Cancer",
    "CABREAST": "Breast Cancer",
    "CACERVIX": "Cervix Cancer",
    "CACOLON": "Colon Cancer",
    "CALUNG": "Lung Cancer",
    "CALYMPH": "Lymph Cancer",
    "CAMELANO": "Melano Cancer",
    "CAOTHER": "Other Cancer",
    "CAPROSTA": "Prostate Cancer",
    "CASKINNM": "Skin Cancer 1",
    "CASKINDK": "SKin Cancer 2",
    "CAUTERUS": "Uterus Cancer"
}

disease_map = {
    "HIBPDX": "High Blood Pressure",
    "ARTHDX": "Arthritis",
    "CHOLDX": "High Cholesterol",
    "OHRTDX": "Other Heart Disease",
    "DIABDX_M18": "Diabetes",
    "ASTHDX": "Asthma",
    "CHDDX": "Coronary Heart Disease",
    "STRKDX": "Stroke",
    "MIDX": "Heart Attack",
    "EMPHDX": "Emphysema",
    "ANGIDX": "Angina",
    "CHBRON31": "Chronic Bronchitis"
}
# ----------------------------------------------------------------------------------------------------------------------------------------------
# 15. PREDICTIVE MODELING (Random Forest Classifier)
#        Predicting Toxicity Tier based on economic and demographic factors
# ----------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "="*80)
print("INITIALIZING PREDICTIVE MODEL: RANDOM FOREST")
print("="*80)

# 1. Define Features (X) and Target (y)
# Selecting the core economic, demographic, and constructed variables
ml_features = ['FAMINC', 'PUBLIC_TOTAL', 'TOTSLF', 'AGELAST', 'COVERAGE_RATIO_PCT', 'CATASTROPHIC_COST', 'PYUNBL42', 'PROBPY42']

# Create a clean dataframe for ML to ensure no NaN values break the model
ml_df = clean_df.dropna(subset=ml_features + ['TOXICITY_TIER']).copy()

X = ml_df[ml_features]
y = ml_df['TOXICITY_TIER']

# 2. Train/Test Split
# We split the data: 80% to train the model, 20% to test its accuracy.
# stratify=y ensures the 80/20 split maintains the same ratio of Severe/Moderate/None cases.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Initialize and Train the Random Forest Model
# Using 'balanced' class_weight because 'None (Fully Adherent)' is likely much more common in the dataset.
# This forces the model to pay extra attention to the rarer 'Severe' and 'Moderate' cases.
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# 4. Make Predictions on the unseen Test data
y_pred = rf_model.predict(X_test)

# 5. Evaluate the Model (Text Output)
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))
# 6. Visualize the Confusion Matrix
# This shows us exactly where the model gets confused (e.g., predicting 'None' when it was actually 'Severe')
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)

plt.title("Confusion Matrix: Random Forest Predictions", fontsize=14, fontweight='bold')
plt.ylabel('Actual Toxicity Tier')
plt.xlabel('Predicted Toxicity Tier')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7. Visualize Feature Importances
# This answers the "Why?": Which variable carried the most weight in deciding the prediction?
importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')

plt.title("Feature Importance: What Drives Financial Toxicity?", fontsize=14, fontweight='bold')
plt.xlabel("Relative Importance Score")
plt.ylabel("Predictive Feature")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 16. ONE-TO-ONE PREDICTION
#        Testing the model on a hypothetical, single patient
# ----------------------------------------------------------------------------------------------------------------------------------------------

print("\n" + "="*80)
print("ONE-TO-ONE PATIENT PREDICTION")
print("="*80)

# Step 1: Define the raw data for your single patient
# Let's imagine a 45-year-old making $30k, receiving $15k in public aid, but paying $4k out-of-pocket
patient_faminc = 30000
patient_public_total = 15000
patient_totslf = 4000
patient_age = 45

# Step 2: Calculate your engineered features using the same logic from earlier
total_known_cost = patient_public_total + patient_totslf
coverage_ratio_pct = (patient_public_total / (total_known_cost + 1e-9)) * 100
catastrophic_cost = 1 if patient_totslf > (0.10 * patient_faminc) else 0

# Step 3: Package this into a DataFrame with the EXACT same columns used for training
new_patient_df = pd.DataFrame({
    'FAMINC': [patient_faminc],
    'PUBLIC_TOTAL': [patient_public_total],
    'TOTSLF': [patient_totslf],
    'AGELAST': [patient_age],
    'COVERAGE_RATIO_PCT': [coverage_ratio_pct],
    'CATASTROPHIC_COST': [catastrophic_cost],
    # 1 = Yes, 2 = No
    'PROBPY42': [2],  # e.g., No problems paying bills
    'PYUNBL42': [2],  # e.g., No unpaid bills
    'CRFMPY42': [1]   # e.g., Yes, currently paying off medical debt over time
})

# Step 4: Make the definitive prediction
tier_prediction = rf_model.predict(new_patient_df)[0]
print(f"Patient Profile:")
print(f" - Income: ${patient_faminc:,} | OOP Cost: ${patient_totslf:,} | Public Aid: ${patient_public_total:,}")
print(f"\n>> PREDICTED OUTCOME: {tier_prediction}")

# Step 5: (Pro-Tip) Get the exact probabilities!
# This tells you how confident the model is, rather than just giving a hard answer.
prediction_probs = rf_model.predict_proba(new_patient_df)[0]
classes = rf_model.classes_

print("\n>> RISK BREAKDOWN (Model Confidence):")
for tier_name, prob in zip(classes, prediction_probs):
    print(f" - {tier_name}: {prob*100:.1f}%")