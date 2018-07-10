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
import charset

def get_dataset(args=None):
    """
    Get a dataset from generator (args currently just for compatibility)
    Format: [text|image|labels] -- types and shapes can be seen below 
    """
    return tf.data.Dataset.from_generator(_generator_wrapper, 
               (tf.string, tf.int32, tf.int32), # Output Types
               (tf.TensorShape([]),             # Text shape
               (tf.TensorShape((32, None, 3))), # Image shape
               (tf.TensorShape([None]))))       # Labels shape

def preprocess_fn(caption, image, labels):
    """Prepare dataset for ingestion"""

    image = _preprocess_image(image)

    # Width is the 2nd element of the image tuple
    width = tf.size(image[1]) 

    # Length is the length of labels - 1
    # (because labels has -1 EOS token here)
    length = tf.subtract(tf.size(labels), -1) 

    text = caption

    return image, width, labels, length, text

def postbatch_fn(image, width, label, length, text):
    # Convert dense to sparse with EOS token of -1
    # Labels must be sparse for ctc_loss
    label = tf.contrib.layers.dense_to_sparse(label, -1)
    
    # Format relevant features for estimator ingestion
    features = {
        "image" : image, 
        "width" : width,
        "length": length,
        "text"  : text
    }

    return features, label

def element_length_fn(image, width, label, length, text):
    return width

def _generator_wrapper():
    """
    Compute the labels in python before everything becomes tensors
    """
    gen = data_generator()
    while True:
        data = next(gen)
        caption = data[0]
        image = data[1]

        # Transform string text to sequence of indices using charset dict
        labels = [charset.out_charset_dict[c] for c in list(caption)]
        
        # Add in -1 as an EOS token for sparsification in postbatch_fn
        labels.append(-1)

        yield caption, image, labels

def _preprocess_image(image):
    # Final image should be pre-grayed in opencv *before* generation
    image = tf.image.rgb_to_grayscale(image) 
    
    image = pipeline.rescale_image(image)

    return image
