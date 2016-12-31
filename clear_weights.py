from tensorflow.core.framework import graph_pb2
import argparse
import numpy as np
import tensorflow as tf

# This strips the weights from a graphdef fiile,
# used for rough measurement the size of the file without weights

parser = argparse.ArgumentParser()

parser.add_argument('file', help='The graph file to further compress')
parser.add_argument('--whitelisted', default='', help='Variables not to convert')

# goes after all constants, except those in whitelisted
def converge_weights(graph_def, whitelisted=[], verbose=True, min_n_weights=None):
        # iterate over all nodes
        for n in graph_def.node:
            # check if right type of node
            if n.op == "Const" and n.name not in whitelisted:
                # replace in
                n.attr['value'].tensor.CopyFrom(tf.contrib.util.make_tensor_proto(np.array([])))

        return graph_def

if __name__ == "__main__":
    args = parser.parse_args()

    graph_def = graph_pb2.GraphDef()
    with open(args.file, "rb") as f:
        graph_def.ParseFromString(f.read())

    new_graph_def = converge_weights(graph_def, whitelisted=args.whitelisted.split(","))

    tf.train.write_graph(new_graph_def, '.', args.file + ".empty", as_text=False)
