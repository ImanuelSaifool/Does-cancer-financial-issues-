import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------
# dataset's entrance into this world
df2021p1 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part1.csv")
df2021p2 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2021_data_part2.csv")
df2022 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2022%20data.csv")
df2023 = pd.read_csv("https://raw.githubusercontent.com/ImanuelSaifool/Does-cancer-financial-issues-/Imanuel's-Test-site/2023%20data.csv")

# renaming stuff
df2021p1 = df2021p1.rename(columns={"TOTSLF21": "TOTSLF"})
df2021p2 = df2021p2.rename(columns={"TOTSLF21": "TOTSLF"}) 
df2022 = df2022.rename(columns={"TOTSLF22": "TOTSLF"})
df2023 = df2023.rename(columns={"TOTSLF23": "TOTSLF"})
df2021p1 = df2021p1.rename(columns={"FAMINC21": "FAMINC"})
df2021p2 = df2021p2.rename(columns={"FAMINC21": "FAMINC"}) 
df2022 = df2022.rename(columns={"FAMINC22": "FAMINC"})
df2023 = df2023.rename(columns={"FAMINC23": "FAMINC"})

# combining them
main_df = pd.concat([df2021p1, df2021p2, df2022, df2023], axis=0)


# data filter
features = ["FAMINC","TOTSLF", "AGELAST", "SEX", "CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO", "CAOTHER"]
clean_df = main_df[main_df['CANCERDX'] == 1].copy() 
clean_df = clean_df[(clean_df[features] >= 0).all(axis=1)]

def is_unable(row):
    # Check all three rounds
    rounds = [row.get('UNABLE31'), row.get('UNABLE42'), row.get('UNABLE53')]
    
    # If ANY round is 1 (Yes), they faced a barrier
    if 1 in rounds:
        return 1
    elif all(r == 2 or pd.isna(r) or r < 0 for r in rounds): 
        return 0
    else:
        return np.nan

clean_df['UNABLE'] = clean_df.apply(is_unable, axis=1)

clean_df = clean_df.dropna(subset=['UNABLE'])
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ==========================================
# FIRST: Summary Statistics
# ==========================================
summary_stats = clean_df[features].describe()
print(summary_stats)

print(clean_df.groupby('UNABLE')['TOTSLF'].mean())

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Set the aesthetic style for your plots
sns.set_style("whitegrid")

# ==========================================
# 1. THE "PROBLEM SOLVER": Box Plot
# ==========================================
plt.figure(figsize=(8, 6))
# We limit Y-axis to 10,000 to zoom in (since max is 169k, it squashes the plot)
sns.boxplot(x='UNABLE', y='TOTSLF', data=clean_df, showfliers=False, palette="Set2")
plt.title('Do Patients Who "Quit" Pay More?', fontsize=14)
plt.ylabel('Out-of-Pocket Cost ($)', fontsize=12)
plt.xlabel('Unable to Access Care (0=No, 1=Yes)', fontsize=12)
plt.show()

# ==========================================
# 2. THE "RISK FACTOR": Bar Chart by Cancer Type
# ==========================================
# First, we need to "melt" the cancer columns to get a single 'Cancer Type' column
cancer_cols = ["CABLADDR", "CABREAST", "CACERVIX", "CACOLON", "CALUNG", "CALYMPH", "CAMELANO"]
# Calculate the % Unable for each cancer type
risk_data = {}
for cancer in cancer_cols:
    # Filter for people who have this cancer (value 1)
    subset = clean_df[clean_df[cancer] == 1]
    # Calculate percentage who were unable
    if len(subset) > 0:
        risk_data[cancer] = subset['UNABLE'].mean() * 100

# Convert to DataFrame for plotting
risk_df = pd.DataFrame(list(risk_data.items()), columns=['Cancer Type', 'Percent Unable'])
risk_df = risk_df.sort_values('Percent Unable', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Percent Unable', y='Cancer Type', data=risk_df, palette="Reds_r")
plt.title('Which Cancer Patients Face the Biggest Barriers?', fontsize=14)
plt.xlabel('Percentage of Patients Unable to Get Care (%)', fontsize=12)
plt.show()

# ==========================================
# 3. THE "BIG PICTURE": Correlation Heatmap
# ==========================================
plt.figure(figsize=(10, 8))
# Select only numerical/binary columns for correlation
corr_cols = ["UNABLE", "TOTSLF", "FAMINC", "AGELAST", "SEX"]
corr_matrix = clean_df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix: What is Linked to "Unable"?', fontsize=14)
plt.show()