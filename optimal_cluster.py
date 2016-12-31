import numpy as np

# the cost function, currently is just MSE (sum of distance from centroid squared for each point)
def _cost(points, start_idx, end_idx):
    if start_idx == end_idx:
        return 0

    current_points = points[start_idx:end_idx]
    mean = np.mean(current_points)
    current_points_min_mean = current_points - mean
    return np.dot(current_points_min_mean, current_points_min_mean)

# function that computes the value if not memoized
def _optimal_cluster_internal(points, start_idx, end_idx, n_clusters, memoization_table=None):
    # done clustering
    if start_idx == end_idx:
        return (0, -1)

    if n_clusters == 0:
        if start_idx != end_idx:
            # not correct configuration
            return (float("inf"), -1)

    # not necessary in theory, but systems hack to go faster
    if n_clusters == 1:
        return (_cost(points, start_idx, end_idx), end_idx)

    # find minimum configuration
    current_min_idx = None
    current_min_cost = None
    for j in xrange(start_idx, end_idx + 1):
        # set j to be the splitting point
        current_cost = _cost(points, start_idx, j) + _optimal_cluster(points, j, end_idx, n_clusters - 1, memoization_table=memoization_table)[0]

        if current_cost < current_min_cost or current_min_cost is None:
            current_min_cost = current_cost
            current_min_idx = j

    return (current_min_cost, current_min_idx)

# internal clustering function that implements memoization
# requires points to be sorted and a memoization_table
# end_idx is constant throughout an entire call
def _optimal_cluster(points, start_idx, end_idx, n_clusters, memoization_table=None):
    key = (start_idx, n_clusters)
    if key not in memoization_table:
        memoization_table[key] = _optimal_cluster_internal(points, start_idx, end_idx, n_clusters, memoization_table=memoization_table)
    return memoization_table[key]

# main entry point
def optimal_cluster(points, n_clusters):
    mtable = {}
    _, current_min_idx = _optimal_cluster(sorted(points), 0, len(points), n_clusters, memoization_table=mtable)

    splits = []
    current_start = 0
    current_clusters = n_clusters - 1
    while current_min_idx != -1:
        splits.append((current_min_idx, np.mean(points[current_start:current_min_idx])))
        current_start = current_min_idx
        _, current_min_idx = mtable[(current_min_idx, current_clusters)]
        current_clusters = current_clusters - 1

    return splits
