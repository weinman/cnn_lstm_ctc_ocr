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
    """
    Reformat raw data for model trainer. 
    Intended to get data as formatted from get_dataset function.
    Parameters:
      caption : tf.string corresponding to text
      image   : tf.int32 tensor of shape [32, ?, 3]
      labels  : tf.int32 tensor of shape [?]
    Returns:
      image   : preprocessed image
                  tf.float32 tensor of shape [32, ?, 1] (? = width)
      width   : width (in pixels) of image
                  tf.int32 tensor of shape []
      labels  : list of indices of characters mapping text->out_charset
                  tf.int32 tensor of shape [?] (? = length+1)
      length  : length of labels (sans -1 EOS token)
                  tf.int32 tensor of shape []
      text    : ground truth string
                  tf.string tensor of shape []
    
    """
    image = _preprocess_image(image)

    # Width is the 2nd element of the image tuple
    width = tf.size(image[1]) 

    # Length is the length of labels - 1
    # (because labels has -1 EOS token here)
    length = tf.subtract(tf.size(labels), -1) 

    text = caption

    return image, width, labels, length, text

def postbatch_fn(image, width, label, length, text):
    """ 
    Prepare dataset for ingestion by Estimator.
    Sparsifies labels, and 'packs' the rest of the components into feature map
    """

    # Convert dense to sparse with EOS token of -1
    # Labels must be sparse for ctc functions (loss, decoder, etc)
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
    """ 
    Determine element length
    Note: mjsynth version of this function has extra parameter (filename)
    """
    return width

def _generator_wrapper():
    """
    Wraps data_generator to precompute labels in python before everything
    becomes tensors. 
    Returns:
      caption : ground truth string
      image   : raw mat object image [32, ?, 3] 
      label   : list of indices corresponding to out_charset 
                length=len(caption)
    """
    gen = data_generator()
    while True:
        data = next(gen)
        caption = data[0]
        image = data[1]

        # Transform string text to sequence of indices using charset dict
        label = [charset.out_charset_dict[c] for c in list(caption)]
        
        # Add in -1 as an EOS token for sparsification in postbatch_fn
        label.append(-1)

        yield caption, image, label

def _preprocess_image(image):
    """Convert image to grayscale and rescale"""
    # Final image should be pre-grayed in opencv *before* generation
    image = tf.image.rgb_to_grayscale(image) 
    
    image = pipeline.rescale_image(image)

    return image
