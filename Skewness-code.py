import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/content/LengthOfStay.csv')
print(df.columns)
stay_data = df['lengthofstay'].dropna()
summary_stats = stay_data.describe()
print(summary_stats)
skewness_value = stay_data.skew()
print("Skewness:", skewness_value)
