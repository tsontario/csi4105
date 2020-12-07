# SOURCE ACKNOWLEDGEMENT: Code based on that taken from https://python-louvain.readthedocs.io/en/latest/
import time

import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

#first compute the best partition

def louvain(G, draw=False):
  # compute the best partition
  start = time.time()
  partition = community_louvain.best_partition(G)
  print(f"Elapsed (n = {G.number_of_nodes()}): {time.time() - start}")
  # draw the graph
  pos = nx.spring_layout(G)
  # color the nodes according to their partition
  cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
  nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
  nx.draw_networkx_edges(G, pos, alpha=0.5)
  plt.show()

node_sizes = [(10, 0.4), (100, 0.2), (500, 0.15), (1000, 0.01), (5000, 0.01), (10000, 0.01)]
for entry in node_sizes:
  G = nx.random_geometric_graph(entry[0], entry[1])
  louvain(G, draw=True)
