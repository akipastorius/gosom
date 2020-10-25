
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import time

data_path = ""
result_path = ""

data = pd.read_csv(data_path, header=None).to_numpy()
result_som = pd.read_csv(result_path, header=None).to_numpy().reshape(-1)

clusters_som = np.unique(result_som)
inspect = pd.DataFrame(np.hstack([data, result_som.reshape(-1, 1)]))
data = data[0:result_som.shape[0], :]

fig = go.Figure()
for cl in clusters_som:
    idx = result_som == cl
    fig.add_trace(go.Scatter(x=data[idx, 0], y=data[idx, 1], name=str(cl), mode='markers'))    

plot(fig)
time.sleep(0.1)

kmeans = KMeans(n_clusters=4, random_state=0).fit(data)

result_kmeans = kmeans.labels_
clusters_kmeans = np.unique(result_kmeans)

fig = go.Figure()
for cl in clusters_kmeans:
    idx = result_kmeans == cl
    fig.add_trace(go.Scatter(x=data[idx, 0], y=data[idx, 1], name=str(cl), mode='markers'))    

plot(fig)


