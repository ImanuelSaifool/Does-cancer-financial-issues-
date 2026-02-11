import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Data's entrance into this teeny weeny world
# ----------------------------------------------------------------------------------------------------------------------------------------------
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2. Standardizing column names
    # we do this so that we can easily integrate multiple data files without changing the name on the raw data
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Out of pocket cost
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"})
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})

# Family income
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"})
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# Insurance covered
df2021p1 = df2021p1.rename(columns={"INSCOV21": "INSCOV"})
df2021p2 = df2021p2.rename(columns={"INSCOV21": "INSCOV"})
df2022 = df2022.rename(columns={"INSCOV22": "INSCOV"})
df2023 = df2023.rename(columns={"INSCOV23": "INSCOV"})

# Combining datasets
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3. FILTERING & CLEANING
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Define feature lists
demog_features = ["FAMINC", "TOTSLF", "AGELAST", "SEX"]
cancer_features = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER"]
features = demog_features + cancer_features 

# Filter for positive cancer diagnosis first
clean_df = main_df[main_df['CANCERDX'] == 1].copy()

# Filter negative values for demographics only to prevent logic error
clean_df = clean_df[(clean_df[demog_features] >= 0).all(axis=1)]

#categorical encoding
    # for other stuff
def is_unable(row):
    val = row.get('PYUNBL42')
    if val == 1: return 1 # Yes, unable to pay
    elif val == 2: return 0 # No, was able to pay
    else: return np.nan

clean_df['UNABLE'] = clean_df.apply(is_unable, axis=1)
clean_df = clean_df.dropna(subset=['UNABLE'])

    #for policy groups
def get_policy_group(code):
    if code == 1: return "Private (Market)"
    elif code == 2: return "Public (Subsidized)"
    elif code == 3: return "Uninsured"
    else: return "Unknown"

clean_df['POLICY_GROUP'] = clean_df['INSCOV'].apply(get_policy_group)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# 4. SUMMARY STATISTICS
# ----------------------------------------------------------------------------------------------------------------------------------------------
sns.set_style("whitegrid")

print("--- GENERAL SUMMARY STATISTICS (General) ---")
print(clean_df[features].describe()) 
print("\n--- Average Out-of-Pocket Cost by Ability to Pay ---")
print(clean_df.groupby('UNABLE')['TOTSLF'].mean())

print("\n--- SUMMARY STATISTICS (For Proposal) ---")
print(clean_df[["TOTSLF", "FAMINC", "UNABLE"]].describe())

print("\n--- THE SUBSIDY SIGNAL: Quitting Rates by Group ---")
policy_stats = clean_df.groupby('POLICY_GROUP')[['UNABLE', 'TOTSLF', 'FAMINC']].mean()
print(policy_stats)