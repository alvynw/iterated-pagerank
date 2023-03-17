import numpy as np

# d = damping factor as described in Brin and Page [1]
d = 0.85

# Directed graph encoding network information
# graph[i][j] = 1 iff page i contains a link to page j
graph = np.matrix([[0, 1, 1, 0, 0],
                   [1, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0]])

print (graph)
