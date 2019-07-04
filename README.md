# Overview

This collection demonstrates how to construct and train a deep,
bidirectional stacked LSTM using CNN features as input with CTC loss
to perform robust word recognition.

The model is a straightforward adaptation of Shi et al.'s CRNN
architecture ([arXiv:1507.0571](https://arxiv.org/abs/1507.05717)).
The provided code downloads and trains using Jaderberg et al.'s
synthetic data ([IJCV 2016](http://dx.doi.org/10.1007/s11263-015-0823-z)), 
MJSynth.

Notably, the model achieves a lower test word error rate (1.82%) than
[CRNN]( https://github.com/bgshih/crnn) when trained and tested on
case-insensitive, closed vocabulary MJSynth data.

Written for Python 2.7. Currently updated for TensorFlow 1.12

The model and subsequent experiments are more fully described in
[Weinman et al. (ICDAR 2019)](https://www.cs.grinnell.edu/~weinman/pubs/weinman19deep.pdf)

# Structure

The model as built is a hybrid of Shi et al.'s CRNN architecture
(arXiv:1507.0571) and the VGG deep convnet, which reduces the number
of parameters by stacking pairs of small 3x3 kernels. In addition, the
pooling is also limited in the horizontal direction to preserve
resolution for character recognition. There must be at least one
horizontal element per character.

Assuming one starts with a 32x32 image, the dimensions at each level
of filtering are as follows:


| Layer |  Op  | KrnSz | Stride(v,h) | OutDim |  H |  W  | PadOpt
|:-----:|------|-------|:-----------:|--------|----|-----|--------------
| 1     | Conv |   3   |   1         |   64   | 30 | 30  |    valid
| 2     | Conv |   3   |   1         |   64   | 30 | 30  |    same
|       | Pool |   2   |   2         |   64   | 15 | 15  | 
| 3     | Conv |   3   |   1         |  128   | 15 | 15  |    same
| 4     | Conv |   3   |   1         |  128   | 15 | 15  |    same
|       | Pool |   2   |   2,1       |  128   |  7 | 14  |       
| 5     | Conv |   3   |   1         |  256   |  7 | 14  |    same
| 6     | Conv |   3   |   1         |  256   |  7 | 14  |    same
|       | Pool |   2   |   2,1       |  256   |  3 | 13  |       
| 7     | Conv |   3   |   1         |  512   |  3 | 13  |    same
| 8     | Conv |   3   |   1         |  512   |  3 | 13  |    same
|       | Pool |   3   |   3,1       |  512   |  1 | 13  |     
| 9     | LSTM |       |             |  512   |    |     |              
| 10    | LSTM |       |             |  512   |    |     |              

To accelerate training, a batch normalization layer is included before
each pooling layer and ReLU non-linearities are used throughout. Other
model details should be easily identifiable in the code.

The default training mechanism uses the ADAM optimizer with learning
rate decay.

## Differences from CRNN

### Deeper early convolutions

The original CRNN uses a single 3x3 convolution in the first two conv/pool
stages, while this network uses a paired sequence of 3x3 kernels. This change
increases the theoretical receptive field of early stages of the network.

As a tradeoff, we omit the computationally expensive 2x2x512 final
convolutional layer of CRNN. In its place, this network vertically max pools over the
remaining three rows of features to collapse to a single 512-dimensional
feature vector at each horizontal location.

The combination of these changes preserves the theoretical receptive field size
of the final CNN layer, but reduces the number of convolution parameters to be
learned by 15%.

### Padding

Another important difference is the lack of zero-padding in the first
convolutional layer, which can cause spurious strong filter responses around
the border. By trimming the first convolution to valid regions, this model
erodes the outermost pixel of values from the response filter maps (reducing
height from 32 to 30 and reducing the width by two pixels).

This approach seems preferable to requiring the network to learn to ignore
strong Conv1 responses near the image edge (presumably by weakening the power
of filters in subsequent convolutional layers).

### Batch normalization

We include batch normalization after each pair of convolutions (i.e., after
layers 2, 4, 6, and 8 as numbered above). The CRNN does not include batch
normalization after its first two convolutional stages. Our model therefore
requires greater computation with an eye toward decreasing the number of
training iterations required to reach converegence.

### Subsampling/stride

The first two pooling stages of CRNN downsample the feature maps with a stride
of two in both spatial dimensions. This model instead preserves sequence length
by downsampling horizontally only after the first pooling stage.

Because the output feature map must have at least one timeslice per character
predicted, overzealous downsampling can make it impossible to represent/predict
sequences of very compact or narrow characters. Reducing the horizontal
downsampling allows this model to recognize words in narrow fonts.

This increase in horizontal resolution does mean the LSTMs must capture more
information. Hence this model uses 512 hidden units, rather than the 256 used
by the CRNN. We found this larger number to be necessary for good performance.

# Training

To completely train the model, you will need to download the mjsynth
dataset and pack it into sharded TensorFlow records. Then you can start
the training process, a tensorboard monitor, and an ongoing evaluation
thread. The individual commands are packaged in the accompanying `Makefile`.

    make mjsynth-download
    make mjsynth-tfrecord
    make train &
    make monitor &
    make test

To monitor training, point your web browser to the url (e.g.,
(http://127.0.1.1:8008)) given by the Tensorboard output.

Note that it may take 4-12 hours to download the complete mjsynth data
set. A very small set (0.1%) of packaged example data is included; to
run the small demo, skip the first two lines involving `mjsynth`.

With a Geforce GTX 1080, the demo takes about 20 minutes for the
validation character error to reach 45% (using the default
parameters); at one hour (roughly 7000 iterations), the validation
error is just over 20%.

With the full training data, by one million iterations the model
typically converges to around 5% training character error and 27.5%
word error.

# Testing

The evaluate script (`src/evaluate.py`) streams statistics for one batch
of validation (or evaluation) data. It prints the iteration, evaluation batch
loss, label error (percentage of characters predicted incorrectly),
and the sequence error (percentage of words—entire sequences—predicted
incorrectly).

The test script (`src/test.py`) tallies statistics, finally
normalizing for all data. It prints the loss, label error, total number of
labels, sequence error, total number of sequences, and the label error
rate and sequence error rate.

# Validation

To see the output of a small set of instances, the script
`validation.py` allows you to load a model and read an image one at a
time via the process's standard input and print the decoded output for
each. For example

    cd src ; python validate.py < ~/paths_to_images.txt

Alternatively, you can run the program interactively by typing image
paths in the terminal (one per line, type Control-D when you want the
model to run the input entered so far).

# Configuration

There are many command-line options to configure training
parameters. Run `train.py` or `test.py` with the `--help` flag to see
them or inspect the scripts. Model parameters are not command-line
configurable and need to be edited in the code (see `src/model.py`).

# Dynamic training data

Dynamic data can be used for training or testing by setting the
`--nostatic_data` flag.

You can use the `--ipc_synth` boolean flag [default=True] to determine
whether to use single-threaded or a buffered, multiprocess synthesis.

The `--synth_config_file` flag must be given with `--nostatic_data`.

The
[MapTextSynthesizer](https://github.com/weinman/MapTextSynthesizer)
library supports training with dynamically synthesized data. The
relevant code can be found within
[MapTextSynthesizer/tensorflow/generator](https://github.com/weinman/MapTextSynthesizer/tree/src/tensorflow/generator)

# Using a lexicon

By default, recognition occurs in "open vocabulary" mode. That is, the
system observes no constraints on producing the resulting output
strings. However, it also has a "closed vocabulary" mode that can
efficiently limit output to a given word list as well as a "mixed
vocabulary" mode that can produce either a vocabulary from a given
word list (lexicon) or a non-vocabulary word, depending on the value
of a prior bias for lexicon words.

Using the closed or mixed vocabulary modes requires additional
software.  This repository is connected with a 
[fork of Harald Scheidl's CTCWordBeamSearch](https://github.com/weinman/CTCWordBeamSearch), obtainable as follows:

```bash
git clone https://github.com/weinman/CTCWordBeamSearch
cd CTCWordBeamSearch
git checkout var_seq_len
```

Then follow the build instructions, which may be as simple as running

```bash
cd cpp/proj
./buildTF.sh
```

To use, make sure `CTCWordBeamSearch/cpp/proj` (the directory
containing `TFWordBeamSearch.so`) is in the `LD_LIBRARY_PATH` when
running `test.py` or `validate.py` (in this repository). 

# API Notes

This version uses the TensorFlow
[Dataset](https://www.tensorflow.org/guide/datasets) for fast
I/O. Training, testing, validation, and prediction use a custom
[Estimator](https://www.tensorflow.org/guide/estimators).

# Citing this work

Please cite the following [paper](https://www.cs.grinnell.edu/~weinman/pubs/weinman19deep.pdf) if you use this code in your own research work:

```text
@inproceedings{ weinman19deep,
    author = {Jerod Weinman and Ziwen Chen and Ben Gafford and Nathan Gifford and Abyaya Lamsal and Liam Niehus-Staab},
    title = {Deep Neural Networks for Text Detection and Recognition in Historical Maps},
    booktitle = {Proc. IAPR International Conference on Document Analysis and Recognition},
    month = {Sep.},
    year = {2019},
    location = {Sydney, Australia}
} 
```

# Acknowledgment

This work was supported in part by the National Science Foundation
under grant Grant Number
[1526350](http://www.nsf.gov/awardsearch/showAward.do?AwardNumber=1526350).
