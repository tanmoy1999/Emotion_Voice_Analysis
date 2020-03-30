
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# Importing the dataset
os.chdir(r'C:\Users\Tanmoy\Desktop\datasci')
dataset = pd.read_csv('emotionData.csv')
X = dataset.iloc[:, [1, 2]].values
# y = dataset.iloc[:, 3].values

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
'''plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')'''
plt.legend()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
import seaborn as sns
# Importing the dataset
os.chdir(r'C:\Users\Tanmoy\Desktop\datasci')
dataset = pd.read_csv('emotionData.csv')
X = dataset.iloc[:,[1,2]].values
#y = dataset.iloc[:, 3].values

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s =11, marker='o', label = 'Vol H')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s =11, marker='s', label = 'Vol A')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 33, c = 'yellow', label = 'Centroids',)
plt.title('K Mean Cluster Analysis')
plt.xlabel('Vol H')
plt.ylabel('Vol A')
plt.legend()
plt.show()

#%%
