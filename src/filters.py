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

# filters.py - Options/examples for building Dataset input filters based on
# image or label string size


# READ THIS BEFORE WRITING YOUR OWN FILTER FUNCTION
#
# To be used as filter_fn's passed into call to pipeline.get_data
# Filters for must be written in the following structure:
# * filter_fn(image, width, label, length, text)
# VAR            : tf.dtype, shape
#--------------------------------------------
# image          : tf.float32, [32, ?, 1] HWC
# width          : tf.int32, []
# label          : tf.int32, [?]
# length         : tf.int32, []
# text           : tf.string, []
#
#
# Note: For more detailed information, refer to `_preprocess_fn`
# in mjsynth.py. Filtering is performed after
# these transformations have been applied. Therefore, filter_fn args
# will correspond to the return values of `preprocess_fn`. 
 
import tensorflow as tf
import model

def input_filter_fn( min_image_width=None, max_image_width=None,
                     min_string_length=None, max_string_length=None,
                     check_input=False):
    """Functor for filter based on string or image size
    Input:
      min_image_width  : Python numerical value (or None) representing the 
                         minimum allowable input image width 
      max_image_width  : Python numerical value (or None) representing the 
                         maximum allowable input image width 
      min_string_length : Python numerical value (or None) representing the 
                         minimum allowable input string length
      max_string_length : Python numerical value (or None) representing the 
                         maximum allowable input string length
      check_input: Whether to verify the feature sequence is as long as the 
                    input string
   Returns:
      keep_input : Boolean Tensor indicating whether to keep a given input 
                  with the specified image width and string length
    """

    if not (min_image_width or max_image_width or
            min_string_length or max_string_length or check_input):
        return None
    
    filter_fn = lambda image, width, label, length, text: \
                _get_filter( width, length,
                             min_image_width, max_image_width,
                             min_string_length, max_string_length,
                             check_input)

    return filter_fn


def _get_filter(width, length, min_width, max_width, min_length, max_length,
                check_input):
    """Function for filter based on string or image size
    Input:
      width      : Tensor representing the image width
      length     : Tensor representing the ground truth string length
      min_width  : Python numerical value (or None) representing the 
                     minimum allowable input image width 
      max_width  : Python numerical value (or None) representing the 
                     maximum allowable input image width 
      min_length : Python numerical value (or None) representing the 
                     minimum allowable input string length
      max_length : Python numerical value (or None) representing the 
                     maximum allowable input string length
      check_input: Whether to verify the feature sequence is as long as the 
                    input string
   Returns:
      keep_input : Boolean Tensor indicating whether to keep a given input 
                     with the specified image width and string length
    """
    
    def add_filter(orig_filt, new_filt): # Helper to build conjunctions
        if orig_filt is None:
            return new_filt
        else:
            return tf.logical_and( orig_filt, new_filt )
        
    keep_input = None

    if min_width:
        keep_input = add_filter( keep_input,
                                 tf.greater_equal(width, min_width) )
    if max_width:
        keep_input = add_filter( keep_input,
                                 tf.less_equal(width, max_width) )
    if min_length:
        keep_input = add_filter( keep_input,
                                tf.greater_equal(length, min_length) )
    if max_length:
        keep_input = add_filter( keep_input,
                                tf.less_equal(length, max_length) )
    if check_input:
        keep_input = add_filter(
            keep_input,
            tf.less_equal( tf.cast(length,tf.int32),
                              model.get_sequence_lengths(width) ) )
        
    if keep_input!=None:
        keep_input = tf.reshape( keep_input, [] ) # explicitly make a scalar

    return keep_input
