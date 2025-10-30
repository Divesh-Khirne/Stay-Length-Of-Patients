import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ------------------------------
# 1. Load CSV dataset
# ------------------------------
df = pd.read_csv('/content/LengthOfStay.csv')

# Rename columns for simplicity
df.rename(columns={'lengthofstay':'LOS', 'gender':'Gender'}, inplace=True)

# Optional: create age groups if age is available; here, BMI can be used for subgroup instead
# Example using BMI groups
df['BMI_Group'] = pd.cut(df['bmi'], bins=[0,18.5,24.9,29.9,100], labels=['Underweight','Normal','Overweight','Obese'])

# ------------------------------
# 2. Descriptive Statistics
# ------------------------------
mean_LOS = df['LOS'].mean()
median_LOS = df['LOS'].median()
mode_LOS = df['LOS'].mode()[0]
std_LOS = df['LOS'].std()
skewness_LOS = stats.skew(df['LOS'])

print("Mean LOS:", mean_LOS)
print("Median LOS:", median_LOS)
print("Mode LOS:", mode_LOS)
print("Std Dev:", std_LOS)
print("Skewness:", skewness_LOS)

# ------------------------------
# 3. Histogram + Density Plot
# ------------------------------
plt.figure(figsize=(10,6))
sns.histplot(df['LOS'], bins=range(int(df['LOS'].min()), int(df['LOS'].max())+2), kde=True, color='skyblue', edgecolor='black')
plt.axvline(mean_LOS, color='red', linestyle='--', label=f'Mean = {mean_LOS:.2f}')
plt.axvline(median_LOS, color='green', linestyle='-.', label=f'Median = {median_LOS}')
plt.axvline(mode_LOS, color='orange', linestyle=':', label=f'Mode = {mode_LOS}')
plt.title('Histogram and Density Plot of Hospital LOS', fontsize=16)
plt.xlabel('Length of Stay (Days)', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# 4. Box Plot with Outliers
# ------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='LOS', data=df, color='lightgreen')
plt.title('Box Plot of Hospital LOS', fontsize=16)
plt.xlabel('Length of Stay (Days)', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------
# 5. Q-Q Plot (Normality Check)
# ------------------------------
plt.figure(figsize=(8,6))
stats.probplot(df['LOS'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Hospital LOS', fontsize=16)
plt.xlabel('Theoretical Quantiles', fontsize=14)
plt.ylabel('Observed LOS Quantiles', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------
# 6. Skewness Illustration
# ------------------------------
x = np.linspace(df['LOS'].min(), df['LOS'].max(), 100)
y_norm = stats.norm.pdf(x, mean_LOS, std_LOS)

plt.figure(figsize=(10,6))
plt.plot(x, y_norm, label='Normal Distribution', color='blue', linestyle='--')
sns.kdeplot(df['LOS'], label='Actual LOS Distribution', color='red')
plt.axvline(mean_LOS, color='red', linestyle='--', label='Mean')
plt.axvline(median_LOS, color='green', linestyle='-.', label='Median')
plt.axvline(mode_LOS, color='orange', linestyle=':', label='Mode')
plt.title('Skewness Illustration: LOS Distribution', fontsize=16)
plt.xlabel('Length of Stay (Days)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# 7. LOS by Gender (Subgroup Analysis)
# ------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='Gender', y='LOS', data=df, palette='Set2')
plt.title('Hospital LOS by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Length of Stay (Days)', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------
# 8. LOS by BMI Group (Subgroup Analysis)
# ------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='BMI_Group', y='LOS', data=df, palette='pastel')
plt.title('Hospital LOS by BMI Group', fontsize=16)
plt.xlabel('BMI Group', fontsize=14)
plt.ylabel('Length of Stay (Days)', fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------
# 9. Outlier Visualization (LOS > 10 days)
# ------------------------------
plt.figure(figsize=(12,4))
outliers = df[df['LOS'] > 10]
plt.scatter(outliers.index, outliers['LOS'], color='red', alpha=0.5, s=10)
plt.title('Outliers in LOS (Patients with LOS > 10 Days)', fontsize=16)
plt.xlabel('Patient Index', fontsize=14)
plt.ylabel('Length of Stay (Days)', fontsize=14)
plt.tight_layout()
plt.show()
