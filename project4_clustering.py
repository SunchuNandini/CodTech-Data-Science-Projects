import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. GENERATE DATASET
np.random.seed(42)
data_size = 300
customer_data = {
    'Age': np.random.randint(18, 70, size=data_size),
    'Total_Spend': np.random.randint(100, 2000, size=data_size)
}
df_clusters = pd.DataFrame(customer_data)

# 2. APPLY K-MEANS CLUSTERING MACHINE LEARNING MODEL
X = df_clusters[['Age', 'Total_Spend']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clusters['Cluster'] = kmeans.fit_predict(X)

# 3. VISUALIZE THE CUSTOMER SEGMENTS
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Total_Spend', hue='Cluster', data=df_clusters, palette='Set1', s=70)
plt.title('Task 4: Customer Segmentation Using K-Means Clustering')
plt.xlabel('Customer Age')
plt.ylabel('Total Spend ($)')
plt.savefig('customer_clusters.png')
plt.close()

print("Project 4: Customer Segmentation (Clustering) executed smoothly!")
