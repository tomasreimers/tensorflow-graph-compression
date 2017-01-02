# TensorFlow Graph Compression

## About
This script was inspired by [Han, Mao, and Dally's work](https://arxiv.org/abs/1510.00149) and implements neural net quantization on TensorFlow graph files.

For a more in-depth explanation, [see here](https://medium.com/@tomasreimers/when-smallers-better-4b54cedc3402#.fwlrwna0b).

## Usage
```
python converge_weights.py FILE_PATH
```

Where `FILE_PATH` is the path to a [GraphDef protobuf](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto) that encodes the graph you want to compress. This will output a `FILE_PATH.min` with is a valid GraphDef protobuf, and can be imported and used in the exact same way as `FILE_PATH`.

Additionally, the script accepts the following flags:

 - `--whitelisted=SomeVariable,SomeOtherVariable`: A comma separated list of constants not to cluster. *Defaults to ''.*
 - `--global_clusters`: Turns on global clusters (one global codebook for weights rather than per layer weights).
 - `--n_clusters=256`: How many clusters to form. *Defaults to 256.*
 - `--min_n_weights=256`: In the case of global clusters, only clusters weights in layers with more than n weights (useful to filter out constants that aren't weights). *Defaults to 256.*

For further configuration options and implementation details, see the code in `converge_weights.py`.

## Author
Tomas Reimers, December 2016

## Copyright and License (MIT License)
Copyright (c) 2016 Tomas Reimers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
