import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

import time

# Attribution notice: code borrows heavily from the following site
# https://towardsdatascience.com/spectral-clustering-aba2640c0d5b

def find_num_clusters(eigenvalues):
  for i, val in enumerate(eigenvalues):
    if val > 0.05:
      return i+1
  return eigenvalues.length()

# For coloring of cluster nodes
colors = [
  "red",
  "lightblue",
  "yellow",
  "orange",
  "purple",
  "green",
  "pink",
  "gray",
  "lightyellow",
  "cyan",
  "lightgreen",
  "white",
  "beige"
]

# Concrete example (small graph)
G_concrete = np.array([
  [0, 1, 1, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [1, 1, 0, 1, 0, 0],
  [0, 0, 1, 0, 1, 1],
  [0, 0, 0, 1, 0, 1],
  [0, 0, 0, 1, 1, 0]])
G = nx.from_numpy_matrix(G_concrete)


### Perform spectral optimization
def spectral(G, draw=False):
  start = time.time()
  # Construct Graph Laplacian
  pos = nx.spring_layout(G)
  A = nx.to_numpy_matrix(G)
  D = np.diag([np.sum(np.asarray(x)) for x in A])
  L = D - A

  # Get eigenvalues and eigenvectors
  eigenvalues, eigenvectors = np.linalg.eig(L)

  # Sort eigenvalues/vectors by ascending order of eigenvalue
  eigenvectors = eigenvectors[:, np.argsort(eigenvalues)]
  eigenvalues = eigenvalues[np.argsort(eigenvalues)]

  if draw:
    for i, val in enumerate(eigenvalues):
      plt.scatter(i+1, np.real(val), c='black')
    plt.xlabel("eigenvalue")
    plt.ylabel("real value")
    plt.show()

  num_clusters = find_num_clusters(eigenvalues)
  print("Num clusters: ", num_clusters)
  if draw is True:
    # Draw graph
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(np.real(eigenvectors[:,1:num_clusters]))
    color_indices = kmeans.labels_
    print(color_indices)
    labels = {}
    for i, node in enumerate(G.nodes()):
      labels[i] = i+1
      nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color=colors[color_indices[i]])
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()
  else:
    print("No draw")

  print("Elapsed: ", time.time() - start)


node_sizes = [10, 100, 500, 1000]
for size in node_sizes:
  G = nx.random_geometric_graph(size,0.4)
  spectral(G, draw=True)
