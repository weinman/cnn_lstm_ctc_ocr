# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
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

# mjsynth.py -- Suite of routines for processing MJSynth data stored
#   as Examples in in TFRecord files.

import os
import tensorflow as tf
import numpy as np
import pipeline

def get_dataset( args ):
    """ Get a Dataset from TFRecord files.
    Parameters:
      base_dir      : Directory containing the TFRecord files
      file_patterns : List of wildcard patterns for TFRecord files to read
      num_threads   : Number of threads to use for reading and processing
      buffer_sz     : Number of Examples to prefetch and buffer
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

    # Extract args
    [ base_dir, file_patterns, num_threads, buffer_sz ] = args[0:4]

    # Get filenames as list of tensors
    tensor_filenames = _get_filenames( base_dir, file_patterns )

    # Get filenames into a dataset format
    ds_filenames = tf.data.Dataset.from_tensor_slices( tensor_filenames )

    # Shuffle for some stochasticity
    ds_filenames = ds_filenames.shuffle( buffer_size=len( tensor_filenames ),
                                         reshuffle_each_iteration=True )
    
    dataset = tf.data.TFRecordDataset( ds_filenames, 
                                       num_parallel_reads=num_threads,
                                       buffer_size=buffer_sz )
    return dataset


def preprocess_fn( data ):
    """Parse the elements of the dataset"""

    feature_map = {
        'image/encoded'  :   tf.FixedLenFeature( [], dtype=tf.string, 
                                                 default_value='' ),
        'image/labels'   :   tf.VarLenFeature( dtype=tf.int64 ), 
        'image/width'    :   tf.FixedLenFeature( [1], dtype=tf.int64,
                                                 default_value=1 ),
        'image/filename' :   tf.FixedLenFeature( [], dtype=tf.string,
                                                 default_value='' ),
        'text/string'    :   tf.FixedLenFeature( [], dtype=tf.string,
                                                 default_value='' ),
        'text/length'    :   tf.FixedLenFeature( [1], dtype=tf.int64,
                                                 default_value=1 )
    }
    
    features = tf.parse_single_example( data, feature_map )
    
    # Initialize fields according to feature map

    # Convert to grayscale
    image = tf.image.decode_jpeg( features['image/encoded'], channels=1 ) 

    width = tf.cast( features['image/width'], tf.int32 ) # for ctc_loss
    label = tf.serialize_sparse( features['image/labels'] ) # for batching
    length = features['text/length']
    text = features['text/string']

    image = preprocess_image( image )

    return image, width, label, length, text


def element_length_fn( image, image_width, label, label_seq_length, text ):
    return image_width


def postbatch_fn( image, width, label, length, text ):
    """Post-batching, postprocessing: packs raw tensors into a dictionary for 
       Dataset's iterator output"""

    # Batching is complete, so now we can re-sparsify our labels for ctc_loss
    label = tf.cast( tf.deserialize_many_sparse( label, tf.int64 ),
                     tf.int32 )
    
    # Format relevant features for estimator ingestion
    features = {
        "image"    : image, 
        "width"    : width,
        "length"   : length,
        "text"     : text
    }

    return features, label


def _get_filenames( base_dir, file_patterns=['*.tfrecord'] ):
    """Get a list of record files"""
    
    # List of lists ...
    data_files = [tf.gfile.Glob( os.path.join( base_dir, file_pattern ) )
                  for file_pattern in file_patterns]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]

    return data_files


def preprocess_image( image ):
    """Preprocess image: Rescale and fix image height"""

    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = pipeline.rescale_image( image )

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice( image, [0, 0, 0], [1, -1, -1] )
    image = tf.concat( [first_row, image], 0 )

    return image
