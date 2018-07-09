# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tensorflow as tf
import numpy as np
from map_generator import data_generator
import pipeline

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset=pipeline.out_charset

def get_dataset(args=None):
    """
    Get a dataset from generator (args currently just for compatibility)
    Format: [text|image|labels] -- types and shapes can be seen below 
    """
    return tf.data.Dataset.from_generator(_generator_wrapper, 
               (tf.string, tf.int32, tf.int32), # Output Types
               (tf.TensorShape([]),             # Shape 1st element
               (tf.TensorShape((32, None, 3))), # Shape 2nd element
               (tf.TensorShape([None]))))       # Shape 3rd element

def preprocess_fn(caption, image, labels):
    """Prepare dataset for ingestion"""

    #NOTE: final image should be pre-grayed by opencv *before* generation
    image = tf.image.rgb_to_grayscale(image) 
    image = _preprocess_image(image)

    # Width is the 2nd element of the image tuple
    width = tf.size(image[1]) 

    # Length is the length of labels - 1 (because labels has -1 EOS token here)
    length = tf.subtract(tf.size(labels), -1) 

    text = caption

    return image, width, labels, length, text

def postbatch_fn(image, width, label, length, text):
    # Convert dense to sparse with EOS token of -1
    label = tf.contrib.layers.dense_to_sparse(label, -1)
    return image, width, label, length, text

def element_length_fn(image, width, label, length, text):
    return width

def _generator_wrapper():
    """
    Compute the labels in python before everything becomes tensors
    Note: Really should not be doing this in python if we don't have to!!!
    """
    gen = data_generator()
    while True:
        data = next(gen)
        caption = data[0]
        image = data[1]

        # Transform string text to sequence of indices using charset
        labels = [out_charset.index(c) for c in list(caption)]
        labels.append(-1)
        yield caption, image, labels

def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    return image
