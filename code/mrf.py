## Python reimplementation of the algorithm described in the paper:

#  Marius Leordeanu and Martial Hebert,
#  Efficient MAP approximation for dense energy functions,
#  International Conference on Machine Learning, May, 2006.


# Utility:

# used for Maximum A Posteriori (MAP) labeling (discrete inference) problems
# where the task is to assign to each node i a label a
# such assignment is represented by a vector x, indexed by ia,
# such that x_{ia} = 1 if node i is labeled with a and 0 otherwise
# the goal is to maximize the labeling score x'Mx + Dx

# Input:

#       M: [N, N] matrix with pairwise potentials
#       D: [N] vector with unary potentials
#       node_indices: [N], node_indices[ia] = i, indicates the node x[ia] corresponds to
#       label_indices: [N], label_indices[ia] = a, indicates the label x[ia] corresponds to
#       iterEigen: nr of iterations of the initial stage (approx 30)
#       iterClimb: nr of iterations of the final stage   (approx 200)

import numpy as np


eps = 1e-6
def mrf(M, D, node_indices, label_indices, iterEigen, iterClimb):

    n = len(M)
    num_nodes = np.max(node_indices) + 1

    x = np.ones(n)

    # the indices in x that correspond to node j
    x_idx = [np.where(node_indices == j)[0] for j in range(num_nodes)]

    ## Stage 1: obtain the starting point using the normalized power / eigen method
    # this finds the global maximum to the relaxed problem

    for _ in range(iterEigen):
        # x = np.matmul(2 * M, x) + D
        x = np.matmul(M, x)
        for j in range(num_nodes):
            x[x_idx[j]] /= np.sqrt((x[x_idx[j]] ** 2).sum() + eps)


    # now start from x, project in on the simplex, and keep climbing using a similar iterative method

    for j in range(num_nodes):
        x[x_idx[j]] /= np.sum(x[x_idx[j]]) + eps

    ## Stage 2: climb until convergence

    step = 1 / iterClimb
    beta = np.arange(1, 0.01, -step)

    iterClimb = len(beta)
    beta = 1 / beta

    for i in range(iterClimb):
        prev_x = x
        for j in range(num_nodes):
            x[x_idx[j]] = prev_x[x_idx[j]] * ((np.matmul(2 * M[x_idx[j]], prev_x) + D[x_idx[j]]) ** beta[i])
            x[x_idx[j]] /= np.sum(x[x_idx[j]]) + eps


    sol = np.zeros(n)
    labels = []
    for j in range(num_nodes):
        index = np.argmax(x[x_idx[j]])
        index = x_idx[j][index]
        sol[index] = 1
        labels.append(label_indices[index])

    score = np.matmul(sol, np.matmul(M, sol)) + np.dot(D, sol)

    return sol, score, labels

















