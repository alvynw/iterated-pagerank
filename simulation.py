import numpy as np
from sklearn.preprocessing import normalize

# References:
# [1] https://research.google/pubs/pub334/ 
# [2] https://disco.ethz.ch/courses/ti2/lecture/markov_chains.pdf
# [3] http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html

# Variables are named as in [2]

# alpha is the damping factor as described in Brin and Page [1]
# 0 < alpha < 1
alpha = 0.85

# Directed graph encoding network information
# To avoid many transposes, I adopt the convetion of:
# graph[j][i] = 1 iff page i contains a link to page j
# This is the opposite of the convention in class
graph = np.array([[0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0]]) 
N = graph.shape[0] 

# Number of iterations of PageRankN to compute
ITERATIONS = 100

print(f"Graph:\n{graph}")

random = np.full((N,), 1/N)
identity = np.identity(N)
zero = np.zeros(N)

pagerank_prev = random
for i in range(0, ITERATIONS):
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

    pagerank_prev = pagerank_N
