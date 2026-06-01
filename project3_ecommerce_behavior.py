import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. GENERATE DATASET
np.random.seed(25)
data_size = 350
ecommerce_data = {
    'Customer_ID': range(2001, 2001 + data_size),
    'Age': np.random.randint(18, 70, size=data_size),
    'Gender': np.random.choice(['Male', 'Female'], size=data_size, p=[0.48, 0.52]),
    'Annual_Income_K': np.random.randint(20, 120, size=data_size),
    'Total_Spend': np.round(np.random.uniform(50.0, 2000.0, size=data_size), 2),
    'Preferred_Category': np.random.choice(['Electronics', 'Clothing', 'Beauty', 'Home'], size=data_size),
    'Satisfaction_Score': np.random.choice([1, 2, 3, 4, 5], size=data_size, p=[0.05, 0.1, 0.2, 0.45, 0.2])
}
df_ecommerce = pd.DataFrame(ecommerce_data)

# 2. DATA VISUALIZATION
sns.set_theme(style="whitegrid")

# Plot 1: Income vs Spend
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual_Income_K', y='Total_Spend', hue='Preferred_Category', data=df_ecommerce, palette='viridis')
plt.title('Customer Segmentation: Income vs Total Spending')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Total Spend ($)')
plt.savefig('income_vs_spend_segments.png')
plt.close()

# Plot 2: Category Spend
plt.figure(figsize=(8, 5))
sns.boxplot(x='Preferred_Category', y='Total_Spend', data=df_ecommerce, palette='pastel')
plt.title('Spending Behavior Across Product Categories')
plt.xlabel('Preferred Product Category')
plt.ylabel('Total Spend ($)')
plt.savefig('category_spending_behavior.png')
plt.close()

print("Project 3: E-Commerce Customer Behavior Analysis executed smoothly!")
