from sklearn.cluster import KMeans
from tensorflow.core.framework import graph_pb2
import argparse
import numpy as np
import tensorflow as tf
import optimal_cluster

## Parameters for KMeans (For documentation, see: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

# can't use k-means++ due to known bug (https://github.com/scikit-learn/scikit-learn/issues/7705)
# problem is fixed in master; will update when patch hits release
km_init = "random"
km_jobs = 2 # -1 to use all CPUs
km_n = 2
km_max_iter = 300

## Whether to use KMeans or Optimal Clustering (see optimal_cluster.py)

USE_KM = True

## Arguments

parser = argparse.ArgumentParser()

parser.add_argument('file', help='The graph file to further compress')
parser.add_argument('--whitelisted', default='', help='Variables not to convert')
parser.add_argument('--global_clusters', action="store_true", help='Whether to apply clustering per const or for the whole graph')
parser.add_argument('--n_clusters', default=256, type=int, help='How many clusters to have')
parser.add_argument('--min_n_weights', default=256, type=int, help='The minimum amount of values in an Const for it to be compressed (helps filter consts that aren\'t weights, only applies to global clustering)')

## Clusterer abstraction

class Clusterer(object):
    def __init__(self):
        self._km = None

    def train(self, n_clusters, values, verbose=False):
        if USE_KM:
            self._km = KMeans(n_clusters=n_clusters, init=km_init, n_init=km_n, n_jobs=km_jobs, max_iter=km_max_iter, verbose=verbose).fit(values)
        else:
            self._centroids = [centroid for _, centroid in optimal_cluster.optimal_cluster(values.flatten(), n_clusters)]

    def predict(self, val):
        if USE_KM:
            return self._km.cluster_centers_[self._km.predict(val)].item(0)
        else:
            idx = (np.abs(self._centroids-val)).argmin()
            return array[self._centroids]

## Weight convergence code (split into global: one codebook for all weights, and local: one codebook per layer)

def _converge_weights_global(graph_def, whitelisted=[], n_clusters=256, verbose=True, min_n_weights=None):
    val_flatten = None

    if verbose:
        print "Collecting weights"

    # iterate over all nodes
    for n in graph_def.node:
        # check if right type of node
        if n.op == "Const" and n.name not in whitelisted:
            # extract values
            val = tf.contrib.util.make_ndarray(n.attr['value'].tensor)

            # don't cluster if it has less than min_n_weights
            if val.size < min_n_weights:
                continue

            # concatenate all the weights into one array
            if val_flatten is None:
                val_flatten = np.expand_dims(val.flatten(), axis=1)
            else:
                val_flatten = np.concatenate((val_flatten, np.expand_dims(val.flatten(), axis=1)))

    # can't cluster if there are fewer points than clusters
    if val_flatten.size <= n_clusters:
        return

    # do clustering
    if verbose:
        print "Performing Clustering"

    c = Clusterer()
    c.train(n_clusters, val_flatten, verbose=verbose)

    # replace elements
    if verbose:
        print "Replacing elements"

    def replace(x):
        return c.predict(x)
    replace_vectorized = np.vectorize(replace)

    for n in graph_def.node:
        # check if right type of node
        if n.op == "Const" and n.name not in whitelisted:
            # extract values
            val = tf.contrib.util.make_ndarray(n.attr['value'].tensor)

            if val.size < min_n_weights:
                continue

            t = val.dtype
            new_val = replace_vectorized(val)
            new_val = new_val.astype(t)

            # replace in
            n.attr['value'].tensor.CopyFrom(tf.contrib.util.make_tensor_proto(new_val))

    return graph_def

def _converge_weights_local(graph_def, whitelisted=[], n_clusters=256, verbose=True):
    # iterate over all nodes
    for n in graph_def.node:
        # check if right type of node
        if n.op == "Const" and n.name not in whitelisted:
            # extract values
            val = tf.contrib.util.make_ndarray(n.attr['value'].tensor)
            val_flatten = np.expand_dims(val.flatten(), axis=1)

            if val_flatten.size <= n_clusters:
                continue

            if verbose:
                print "Converting: ", n.name, "(", val_flatten.size, " weights)"

            # do kmeans
            if verbose:
                print "Finding cluster centers"

            c = Clusterer()
            c.train(n_clusters, val_flatten, verbose=verbose)

            # replace elements
            if verbose:
                print "Replacing"

            def replace(x):
                return c.predict(x)

            replace_vectorized = np.vectorize(replace)

            t = val.dtype
            new_val = replace_vectorized(val)
            new_val = new_val.astype(t)

            # replace in
            n.attr['value'].tensor.CopyFrom(tf.contrib.util.make_tensor_proto(new_val))

    return graph_def

## Code entry point (to be imported in other modules)

# attempts to replace all constants, except those in whitelisted
def converge_weights(graph_def, whitelisted=[], global_clusters=False, n_clusters=256, verbose=True, min_n_weights=None):
    if global_clusters:
        return _converge_weights_global(graph_def, whitelisted=whitelisted, n_clusters=n_clusters, verbose=True, min_n_weights=min_n_weights)
    else:
        return _converge_weights_local(graph_def, whitelisted=whitelisted, n_clusters=n_clusters, verbose=True)

## Script entry point

if __name__ == "__main__":
    args = parser.parse_args()

    graph_def = graph_pb2.GraphDef()
    with open(args.file, "rb") as f:
        graph_def.ParseFromString(f.read())

    new_graph_def = converge_weights(graph_def, whitelisted=args.whitelisted.split(","), global_clusters=args.global_clusters, n_clusters=args.n_clusters, min_n_weights=args.min_n_weights)

    tf.train.write_graph(new_graph_def, '.', args.file + ".min", as_text=False)
