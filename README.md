# Overview

This collection demonstrates how to construct and train a deep,
bidirectional stacked LSTM using a CNN features as input with CTC loss
to perform robust word recognition. The model is a straightforward
adaptation of Shi et al.'s CRNN architecture
([arXiv:1507.0571](https://arxiv.org/abs/1507.05717)). The provided code
downloads and trains using Jaderberg et al.'s synthetic data ([IJCV
2016](http://dx.doi.org/10.1007/s11263-015-0823-z)).

Developed for Tensorflow 1.1


# Structure

The model as build is a hybrid of Shi et al.'s CRNN architecture
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

# Training

To completely train the model, you will need to download the mjsynth
dataset, pack it into sharded tensorflow records. Then you can start
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
typically converges to around 7% training character error and 35% word
error, both varying by 2-5%.

# Testing

The test script streams statistics for small batches of validation (or
test) data. It ouputs the label error (percentage of characters
predicted incorrectly), the test loss, and the sequence error
(percentage of words--entire sequences--predicted incorrectly.)

# Configuration

There are many command-line options to configure training
parameters. Run `train.py` or `test.py` with the `--help` flag to see
them or inspect the scripts. Model parameters are not command-line
configurable and need to be edited in the code (see `model.py`).
