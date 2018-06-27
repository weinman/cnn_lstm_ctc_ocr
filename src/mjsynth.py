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
#from map_generator import data_generator

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
out_charset_tf=tf.string_split([tf.constant("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")], "")

def num_classes():
    return len(out_charset)

def bucketed_input_pipeline(base_dir,file_patterns,
                            num_threads=4,
                            batch_size=32,
                            boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                            input_device=None,
                            width_threshold=None,
                            length_threshold=None,
                            num_epoch=None):
    """Get input tensors bucketed by image width
    Returns:
      image : float32 image tensor [batch_size 32 ? 1] padded to batch max width
      width : int32 image widths (for calculating post-CNN sequence length)
      label : Sparse tensor with label sequences for the batch
      length : Length of label sequence (text length)
      text  :  Human readable string for the image
      filename : Source file path
    """
    # Get filenames into a dataset format
    filenames = tf.data.Dataset.from_tensor_slices(
        _get_filenames(base_dir, file_patterns))
    
    with tf.device(input_device): # Create bucketing batcher
        
        # https://www.tensorflow.org/performance/datasets_performance
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
                                                cycle_length=num_threads,  
                                                sloppy=True))
        
        # Preprocess
        dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
        
        # Filter out inappropriately dimension-ed elements
        if(width_threshold != None or length_threshold != None):
            dataset = dataset.filter(
                lambda image, width, label, length, text, filename:
                _get_input_filter(width, width_threshold,
                                  length, length_threshold))


        # Bucket according to image width and batch
        dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
            element_length_func=_element_length_fn,
            bucket_batch_sizes=np.full(len(boundaries) + 1, batch_size),
            bucket_boundaries=boundaries))

        # Repeat for num_epochs
        dataset = dataset.repeat(num_epoch)

        # Deserialize sparse tensor
        dataset = dataset.map(
            lambda image, width, label, length, text, filename: 
            (image, 
             width, 
             tf.cast(tf.deserialize_many_sparse(label, tf.int64), 
                     tf.int32),
             length, 
             text, 
             filename),
            num_parallel_calls=num_threads)
        
    return dataset

def threaded_input_pipeline(base_dir,file_patterns,
                            num_threads=4,
                            batch_size=32,
                            batch_device=None,
                            preprocess_device=None):

    # Get filenames into a dataset format
    #filenames = tf.data.Dataset.from_tensor_slices(
    #    _get_filenames(base_dir, file_patterns))

    dataset = tf.data.TFRecordDataset(_get_filenames(base_dir, file_patterns))

    with tf.device(preprocess_device):

        # https://www.tensorflow.org/performance/datasets_performance
        #dataset = filenames.apply(
        #    tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
        #                                        cycle_length=num_threads,  
        #                                        sloppy=True))
        dataset = dataset.map(_parse_function,
                              num_parallel_calls=num_threads)

    
    with tf.device(batch_device): # Create batch

        # Hack -- probably a better way to do this! Just want dynamic padding!
        # Pad batches to max data size (bucketing it all into the same bucket)
        dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length
                                (element_length_func=_element_length_fn,
                                 bucket_batch_sizes=[batch_size, batch_size],
                                 bucket_boundaries=[0]))

        dataset = dataset.map(lambda image, 
                              width, label, 
                              length, text, filename: 
                              (image, width, 
                               tf.cast(tf.deserialize_many_sparse(label, tf.int64), 
                                       tf.int32),
                               length, text, filename))
        num_epochs = None
        # Repeat for num_epochs
        dataset = dataset.repeat(num_epochs)

    return dataset

def _element_length_fn(image, width, label, length, text, filename):
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

def _get_filenames(base_dir, file_patterns=['*.tfrecord']):
    """Get a list of record files"""
    
    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir,file_pattern))
                  for file_pattern in file_patterns]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]

    return data_files


# Note: Currently not in use: probably more optimal than current implmntation
def _text_to_labels(text):
    """Convert given text (tf.string) into a list of tf.int32's"""
    labels = tf.data.Dataset.from_tensor_slices(tf.string_split([text],""))
    
    # Converts ['A', 'B', 'C'] -> [0, 1, 2]
    # Note: MUST RUN tf.tables_initializer().run() in order for this to work
    table = tf.contrib.lookup.index_table_from_tensor(mapping=out_charset_tf,
                                                      num_oov_buckets=1,
                                                      default_value=-1)
    labels = table.lookup(labels)
    
    return labels

def _preprocess_dataset(caption, image, labels):
    """Get everything how it should be"""

    #NOTE: final image should be pre-grayed by opencv *before* generation
    image = tf.image.rgb_to_grayscale(image) 
    image = _preprocess_image(image)

    width = tf.size(image[1]) 
    # labels = _text_to_labels(caption) Not necessary with precomputed labels
    length = tf.size(labels)
    text = caption
    return image, width, labels, length, text
                
"""
def _parse_function(caption, image, labels):
    Parse the elements of the dataset
    
    # Format elements appropriately
    dataset = .map(_preprocess_dataset, num_parallel_calls=num_threads)

    return dataset
"""

def _char_to_int(character):
    """Convert given character (really tf.string of length 1) to its integer representation from out_charset"""
    
    tf.contrib.lookup.string_to_index(character, out_charset_tf) # default val -1
    return out_charset.index(character)

def _generator_wrapper():
    """Compute the labels in python before everything becomes tensors
       Note: very! SUBOPTIMAL-- Really should not be doing this in python
       if we don't have to!!!"""
    gen = data_generator()
    while True:
        data = next(gen)
        caption = data[0]
        image = data[1]

        # Transform string text to sequence of indices using charset
        labels = [out_charset.index(c) for c in list(caption)]
        yield caption, image, labels

def _parse_function(data):
    """Parse the elements of the dataset"""

    feature_map = {
        'image/encoded'  :   tf.FixedLenFeature([], dtype=tf.string, 
                                                default_value='' ),
        'image/labels'   :   tf.VarLenFeature( dtype=tf.int64 ), 
        'image/width'    :   tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=1 ),
        'image/filename' :   tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='' ),
        'text/string'    :   tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='' ),
        'text/length'    :   tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=1 )
    }
    
    features = tf.parse_single_example(data, feature_map)
    
    # Initialize fields according to feature map
    image = tf.image.decode_jpeg( features['image/encoded'], channels=1 ) #gray
    width = tf.cast( features['image/width'], tf.int32) # for ctc_loss
    label = tf.serialize_sparse( features['image/labels'] ) # for batching
    length = features['text/length']
    text = features['text/string']
    filename = features['image/filename']

    # Prepare image
    image = _preprocess_image(image)

    return image,width,label,length,text,filename

def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image
