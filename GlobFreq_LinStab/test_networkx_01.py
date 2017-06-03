import numpy as np
import networkx as nx

n = 3

g = nx.grid_2d_graph(n, n)
g1 = nx.convert_node_labels_to_integers(g, ordering='sorted')



n_osci = n**2
m = np.zeros((n_osci, n_osci))

for i in range(n_osci):
    i_neigh = g1.neighbors(i)
    m[i, i_neigh] = 1
    m[i, :] = m[i, :] / np.sum(m[i, :])