
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.offline import plot

import numpy as np

# Make features and targets with 500 samples,
data, target = make_blobs(n_samples = 500,
                  # two feature variables,
                  n_features = 2,
                  # four clusters,
                  centers = 4,
                  # with .65 cluster standard deviation,
                  cluster_std = 0.65,
                  # shuffled,
                  shuffle = True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers'))
plot(fig)

kmeans = KMeans(n_clusters=4, random_state=0).fit(data)

np.savetxt("data/data.csv", data, delimiter=",")
np.savetxt("data/target.csv", target, delimiter=",")
