'''10. Gene Expression Analysis  
Scenario: In a biological study, researchers have collected gene expression data from thousands of genes across multiple samples
Question: How would PCA help in identifying key patterns or variations in gene expression? What challenges might arise when interpreting the principal components?'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('gene_expression.csv')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Gene Expression Data')
plt.show()
