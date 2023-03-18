import numpy as np
from sklearn.preprocessing import normalize

# References:
# [1] 
# [2]

# Variables are named as in [2]

# alpha is the damping factor as described in Brin and Page [1]
# 0 < alpha < 1
alpha = 0.85

# Directed graph encoding network information
# To avoid transposes, I adopt the convetion of:
# graph[j][i] = 1 iff page i contains a link to page j
# This is the opposite of the convention in class
graph = np.array([[0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0]]) 
N = graph.shape[0] 

ITERATIONS = 5

print(f"Graph:\n{graph}")

# Build initial surfer Markov chain
random = np.full((N, N), 1/N)
#print (random)
surfer_mc0 = alpha*normalize(graph, axis=0, norm="l1") + (1-alpha)*random
#print(normalize(graph, axis=0, norm="l1"))

print (surfer_mc0)

#Solve for PageRank0
identity = np.identity(N)
zero = np.zeros(N)

#print(np.append(zero, [1]))
#print(np.vstack((surfer_mc0 - identity, np.ones(N))))

# Solve for stationary distribtuion
# This encodes: 1) (M - I)x = 0 and 2) entries of x sum to 1
pagerank0, _, _, _ = np.linalg.lstsq(np.vstack((surfer_mc0 - identity, np.ones(N))), np.append(zero, [1]), rcond = None)

print(f"Pagerank0: {pagerank0}")

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
    M_N = alpha * W_N + (1 - alpha) * pagerank_prev_col

    pagerank_N, _, _, _ = np.linalg.lstsq(np.vstack((M_N - identity, np.ones(N))), np.append(zero, [1]), rcond = None)
    print(f"PageRank{i}: {pagerank_N}")

    pagerank_prev = pagerank_N







