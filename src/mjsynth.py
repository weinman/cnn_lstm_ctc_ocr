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

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

def num_classes():
    return len(out_charset)

def get_strings(labels):
    """
    Transform a SparseTensorValue matrix of labels into the corresponding
    list of character strings
    """
    num_strings = labels.dense_shape[0]
    
    strings = []

    for row in range(num_strings):

        indices = np.where( labels.indices[:,0]==row )[0]
        indices.shape = (indices.shape[0], 1)

        label = labels.values[indices]
        label.shape = (label.shape[0],)

        strings.append( get_string( label ))

    return strings

def get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([out_charset[c] for c in labels])
    return string

def bucketed_input_pipeline(base_dir,file_patterns,
                            num_threads=4,
                            batch_size=32,
                            boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                            input_device=None,
                            width_threshold=None,
                            length_threshold=None,
                            num_epochs=None):
    """Get input tensors bucketed by image width
    Returns:
      image : float32 image tensor [batch_size 32 ? 1] padded to batch max width
      width : int32 image widths (for calculating post-CNN sequence length)
      label : Sparse tensor with label sequences for the batch
      length : Length of label sequence (text length)
      text  :  Human readable string for the image
      filename : Source file path
    """
    queue_capacity = num_threads*batch_size*2
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs!=None) 

    data_queue = _get_data_queue(base_dir, file_patterns, 
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs)

    with tf.device(input_device): # Create bucketing batcher
        image, width, label, length, text, filename  = _read_word_record(
            data_queue)
        image = _preprocess_image(image) # move after batch?

        keep_input = _get_input_filter(width, width_threshold,
                                       length, length_threshold)
        data_tuple = [image, label, length, text, filename]
        width,data_tuple = tf.contrib.training.bucket_by_sequence_length(
            input_length=width,
            tensors=data_tuple,
            bucket_boundaries=boundaries,
            batch_size=batch_size,
            capacity=queue_capacity,
            keep_input=keep_input,
            allow_smaller_final_batch=final_batch,
            dynamic_pad=True)
        [image, label, length, text, filename] = data_tuple
        label = tf.deserialize_many_sparse(label, tf.int64) # post-batching...
        label = tf.cast(label, tf.int32) # for ctc_loss
    return image, width, label, length, text, filename

def threaded_input_pipeline(base_dir,file_patterns,
                            num_threads=4,
                            batch_size=32,
                            batch_device=None,
                            preprocess_device=None,
                            num_epochs=None):

    queue_capacity = num_threads*batch_size*2
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs!=None) 

    data_queue = _get_data_queue(base_dir, file_patterns, 
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs)

    # each thread has a subgraph with its own reader (sharing filename queue)
    data_tuples = [] # list of subgraph [image, label, width, text] elements
    with tf.device(preprocess_device):
        for _ in range(num_threads):
            image, width, label, length, text, filename  = _read_word_record(
                data_queue)
            image = _preprocess_image(image) # move after batch?
            data_tuples.append([image, width, label, length, text, filename])

    with tf.device(batch_device): # Create batch queue
        image, width, label, length, text, filename  = tf.train.batch_join( 
            data_tuples, 
            batch_size=batch_size,
            capacity=queue_capacity,
            allow_smaller_final_batch=final_batch,
            dynamic_pad=True)
        label = tf.deserialize_many_sparse(label, tf.int64) # post-batching...
        label = tf.cast(label, tf.int32) # for ctc_loss
    return image, width, label, length, text, filename

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

def _get_data_queue(base_dir, file_patterns=['*.tfrecord'], capacity=2**15,
                    num_epochs=None):
    """Get a data queue for a list of record files"""

    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir,file_pattern))
                  for file_pattern in file_patterns]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]
    data_queue = tf.train.string_input_producer(data_files, 
                                                capacity=capacity,
                                                num_epochs=num_epochs)
    return data_queue

def _read_word_record(data_queue):

    reader = tf.TFRecordReader() # Construct a general reader
    key, example_serialized = reader.read(data_queue) 

    feature_map = {
        'image/encoded':  tf.FixedLenFeature( [], dtype=tf.string, 
                                              default_value='' ),
        'image/labels':   tf.VarLenFeature( dtype=tf.int64 ), 
        'image/width':    tf.FixedLenFeature( [1], dtype=tf.int64,
                                              default_value=1 ),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='' ),
        'text/string':     tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='' ),
        'text/length':    tf.FixedLenFeature( [1], dtype=tf.int64,
                                              default_value=1 )
    }
    features = tf.parse_single_example( example_serialized, feature_map )

    image = tf.image.decode_jpeg( features['image/encoded'], channels=1 ) #gray
    width = tf.cast( features['image/width'], tf.int32) # for ctc_loss
    label = tf.serialize_sparse( features['image/labels'] ) # for batching
    length = features['text/length']
    text = features['text/string']
    filename = features['image/filename']
    return image,width,label,length,text,filename

def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image
