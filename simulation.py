import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import sys

# References:
# [1] https://disco.ethz.ch/courses/ti2/lecture/markov_chains.pdf
# [2] http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# [3] https://research.google/pubs/pub334/ 

# Variables are named as in [1]

# alpha is the damping factor as described in Brin and Page [3]
# 0 < alpha < 1
alpha = 0.85

# Directed graph encoding network information
# To avoid many transposes, I adopt the convetion of:
# graph[j][i] = 1 iff page i contains a link to page j
# This is the opposite of the convention in class

# Graph of Figure 11.21 in [1]
#N = 5
#graph = np.array([[0, 1, 0, 0, 0],
#                  [1, 0, 0, 0, 0],
#                  [1, 1, 0, 0, 0],
#                  [0, 0, 1, 0, 1],
#                  [0, 0, 0, 0, 0]]) 


# Supply your own graph to test
# The format of the data must be:
# # Comments preceded by #
# A line containing i j\n for an edge from j to i
N = 1000
graph = np.zeros((N, N))

with open('data.txt', 'r') as file:
    for line in file:
        if line[0] == '#':
            continue
        [i,j] = [int(x) for x in line.split()]
        if i != j:
            graph[i][j] = 1

# Number of iterations of PageRankN to compute
ITERATIONS = 100

#print(f"Graph:\n{graph}")

random = np.full((N,), 1/N)
identity = np.identity(N)
zero = np.zeros(N)

pagerank_prev = random

#print(f"PageRank0: {random}")

# Iterated PageRank values
pageranks = []
pageranks.append(pagerank_prev)

# Distance between consecutive PageRank values
distances = []

for i in range(1, ITERATIONS + 1):
    pagerank_prev_col = pagerank_prev.reshape(-1, 1)
    W_N = normalize(graph * pagerank_prev_col, axis=0, norm='l1')
   
    # pagerank_prev_col is implicitly broadcast by numpy to be N concatenations of pagerank_prev_col
    # i.e. [pagerank_prev_col ... pagerank_prev_col]
    # which is precisely R_N
    # See: https://numpy.org/doc/stable/user/basics.broadcasting.html for more information on broadcasting
    M_N = alpha * W_N + (1 - alpha) * pagerank_prev_col
    
    # Solve (M_N - I)x = 0 subject to the constraint that the entries of x sum to 1
    pagerank_N, _, _, _ = np.linalg.lstsq(np.vstack((M_N - identity, np.ones(N))), np.append(zero, [1]), rcond = None)
    print(f"PageRank{i}: {pagerank_N}")

    pageranks.append(pagerank_N)
    distances.append(np.linalg.norm(pagerank_N - pagerank_prev))

    pagerank_prev = pagerank_N

print(f"PageRank{ITERATIONS} and PageRank1 distance: {np.linalg.norm(pageranks[ITERATIONS]-pageranks[1])}")
plt.plot(distances)
plt.xlabel("Iteration")
plt.ylabel("Change in Euclidean Distance")
plt.show()
