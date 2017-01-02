from tensorflow.core.framework import graph_pb2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# This strips the weights from a graphdef fiile,
# used for rough measurement the size of the file without weights

parser = argparse.ArgumentParser()

parser.add_argument('file', help='The graph file to further compress')
parser.add_argument('--whitelisted', default='', help='Variables not to count')
parser.add_argument('--graph_title', default='Net', help='name of the net (for plot title)')

# goes after all constants, except those in whitelisted
def plot_weights(graph_def, whitelisted=[], verbose=True, graph_title="Net"):
    val_flatten = None

    if verbose:
        print "Collecting weights"

    # iterate over all nodes
    for n in graph_def.node:
        # check if right type of node
        if n.op == "Const" and n.name not in whitelisted:
            # extract values
            val = tf.contrib.util.make_ndarray(n.attr['value'].tensor)

            # concatenate all the weights into one array
            if val_flatten is None:
                val_flatten = np.expand_dims(val.flatten(), axis=1)
            else:
                val_flatten = np.concatenate((val_flatten, np.expand_dims(val.flatten(), axis=1)))

    # mu = np.mean(val_flatten)
    # sigma = np.std(val_flatten)
    #
    # plt.hist(val_flatten, bins=50, range=(mu - 0.25 * sigma, mu + 0.25 * sigma))
    plt.hist(val_flatten, bins=50, range=(np.percentile(val_flatten, 1), np.percentile(val_flatten, 99)))
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Frequency of Various Values for Weights in {}'.format(graph_title))
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()

    graph_def = graph_pb2.GraphDef()
    with open(args.file, "rb") as f:
        graph_def.ParseFromString(f.read())

    plot_weights(graph_def, whitelisted=args.whitelisted.split(","), graph_title=args.graph_title)
