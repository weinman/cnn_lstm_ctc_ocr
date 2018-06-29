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

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Get a sparse tensor for table lookup in the tensorflow runtime
out_charset_tf=tf.string_split([tf.constant(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")], "")

def num_classes():
    return len(out_charset)

def bucketed_input_pipeline(base_dir=None,file_patterns=None,
                            num_threads=4,
                            batch_size=32,
                            boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                            input_device=None,
                            num_epoch=None,
                            filter_fn=None):
    """Get input dataset with elements bucketed by image width
    Returns:
      image  : float32 image tensor [batch_size 32 ? 1] padded 
                 to max width in batch
      width  : int32 image widths (for calculating post-CNN sequence length)
      label  : Sparse tensor with label sequences for the batch
      length : Length of label sequence (text length)
      text   : Human readable string for the image
    """

    dataset = _get_dataset()

    with tf.device(input_device): # Create bucketing batcher

        dataset = dataset.map(_preprocess_dataset, 
                              num_parallel_calls=num_threads)
        
        # Remove input that doesn't fit necessary specifications
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        # Bucket and batch appropriately
        dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                      element_length_func=_element_length_fn,
                      bucket_batch_sizes=np.full(len(boundaries)+1, batch_size),
                      bucket_boundaries=boundaries))

        # Convert labels to sparse tensor for CNN purposes
        dataset = dataset.map(
            lambda image, width, label, length, text:
                (image, 
                 width, 
                 tf.contrib.layers.dense_to_sparse(label,-1),
                 length, text),
            num_parallel_calls=num_threads).prefetch(1)

    return dataset

def threaded_input_pipeline(base_dir=None,file_patterns=None,
                            num_threads=4,
                            batch_size=32,
                            batch_device=None,
                            preprocess_device=None):

    dataset = _get_dataset()

    with tf.device(preprocess_device):
        dataset = dataset.map(_preprocess_dataset,
                              num_parallel_calls=num_threads)
    
    with tf.device(batch_device): # Create batch

        # Hack -- probably a better way to do this! Just want dynamic padding!
        # Pad batches to max data size (bucketing it all into the same bucket)
        dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                       element_length_func=_element_length_fn,
                       bucket_batch_sizes=[batch_size, batch_size],
                       bucket_boundaries=[0]))

        # Convert to sparse tensor for CNN purposes
        dataset = dataset.map(
            lambda image, width, label, length, text: 
            (image, 
             width, 
             tf.contrib.layers.dense_to_sparse(label,-1),
             length, 
             text),
            num_parallel_calls=num_threads)

        dataset = dataset.prefetch(1)

    return dataset.prefetch(1)

def _element_length_fn(image, width, label, length, text):
    return width

def dataset_element_length_fn(_, image):
    return tf.shape(image)[2]

def _get_input_filter(width, width_threshold, length, length_threshold):
    """Boolean op for discarding input data based on string or image size
    Input:
      width            : Tensor representing the image width
      width_threshold  : Python numerical value (or None) representing the 
                         maximum allowable input image width 
      length           : Tensor representing the ground truth string length
      length_threshold : Python numerical value (or None) representing the 
                         maximum allowable input string length
   Returns:
      keep_input : Boolean Tensor indicating whether to keep a given input 
                  with the specified image width and string length
"""

    keep_input = None

    if width_threshold!=None:
        keep_input = tf.less_equal(width, width_threshold)

    if length_threshold!=None:
        length_filter = tf.less_equal(length, length_threshold)
        if keep_input==None:
            keep_input = length_filter 
        else:
            keep_input = tf.logical_and( keep_input, length_filter)

    if keep_input==None:
        keep_input = True
    else:
        keep_input = tf.reshape( keep_input, [] ) # explicitly make a scalar

    return keep_input

def _get_dataset():
    """
    Get a dataset from generator
    Format: [text|image|labels] -- types and shapes can be seen below 
    """
    return tf.data.Dataset.from_generator(_generator_wrapper, 
               (tf.string, tf.int32, tf.int32), # Output Types
               (tf.TensorShape([]),             # Shape 1st element
               (tf.TensorShape((32, None, 3))), # Shape 2nd element
               (tf.TensorShape([None]))))       # Shape 3rd element

# Note: Currently not in use: probably more optimal than current implmntation
def _text_to_labels(text): #TODO TEST
    """Convert given text (tf.string) into a list of tf.int32's"""
    labels = tf.data.Dataset.from_tensor_slices(tf.string_split([text],""))
    
    # Converts ['A', 'B', 'C'] -> [0, 1, 2]
    # Note: MUST RUN tf.tables_initializer().run() in order for this to work
    table = tf.contrib.lookup.index_table_from_tensor(mapping=out_charset_tf,
                num_oov_buckets=1,
                default_value=0)
    labels = table.lookup(labels)
    return labels

def _preprocess_dataset(caption, image, labels):
    """Prepare dataset for ingestion"""

    #NOTE: final image should be pre-grayed by opencv *before* generation
    image = tf.image.rgb_to_grayscale(image) 
    image = _preprocess_image(image)

    width = tf.size(image[1]) 
    # labels = _text_to_labels(caption) Not necessary with precomputed labels
    # labels = tf.contrib.layers.dense_to_sparse(labels,0) if possible
    length = tf.size(labels)
    text = caption
    return image, width, labels, length, text

def _generator_wrapper():
    """
    Compute the labels in python before everything becomes tensors
    Note: very! SUBOPTIMAL-- Really should not be doing this in python
    if we don't have to!!!
    """
    gen = data_generator()
    while True:
        data = next(gen)
        caption = data[0]
        image = data[1]

        # Transform string text to sequence of indices using charset
        labels = [out_charset.index(c) for c in list(caption)]
        yield caption, image, labels

def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    return image
