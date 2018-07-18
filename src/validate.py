# CNN-LSTM-CTC-OCR
# Copyright (C) 2017, 2018 Jerod Weinman
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
import sys

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn

import pipeline
import model
from lexicon import dictionary_from_file

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )
tf.app.flags.DEFINE_string( 'device','/gpu:0',
                            """Device for graph placement""" )
tf.app.flags.DEFINE_string( 'lexicon','',
			    """File containing lexicon of image words""" )

tf.logging.set_verbosity( tf.logging.WARN )

def _get_image( filename ):
    """Load image data for placement in graph"""

    image = Image.open( filename ) 
    image = np.array( image )
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    image = image[:,:,:1] # so just extract first channel, preserving 3D shape

    return image


def _preprocess_image( image ):

    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype( image, tf.float32 )
    image = tf.subtract( image, 0.5 )

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice( image, [0, 0, 0], [1, -1, -1] )
    image = tf.concat( [first_row, image], 0 )

    return image


def _get_input():
    """Create a dataset of images by reading from stdin"""

    # Initializing the dataset with one image
    image_data = _get_image( raw_input().rstrip() )
    dataset = tf.data.Dataset.from_tensors( image_data )

    # Add the rest of the images to the dataset (if any)
    for line in sys.stdin:
        image_data = _get_image(line.rstrip())
        temp_dataset = tf.data.Dataset.from_tensors( image_data )
        dataset = dataset.concatenate( temp_dataset )
    
    # Iterate over the dataset to extract each image
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    return image

def _get_output( rnn_logits,sequence_length ):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
	if FLAGS.lexicon:
	    dict_tensor = _get_dictionary_tensor( FLAGS.lexicon, pipeline.out_charset )
	    predictions,_ = tf.nn.ctc_beam_search_decoder_trie( rnn_logits,
                                                                sequence_length,
                                                                alphabet_size=
                                                                pipeline.num_classes() ,
                                                                dictionary=dict_tensor,
                                                                beam_width=128,
                                                                top_paths=1,
                                                                merge_repeated=True )
	else:
	    predictions,_ = tf.nn.ctc_beam_search_decoder( rnn_logits,
                                                           sequence_length,
                                                           beam_width=128,
                                                           top_paths=1,
                                                           merge_repeated=True )
    return predictions


def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    return config


def _get_string( labels ):
    """Transform an 1D array of labels into the corresponding character string"""

    string = ''.join( [pipeline.out_charset[c] for c in labels] )
    return string

def _get_dictionary_tensor( dictionary_path, charset ):
    return tf.sparse_tensor_to_dense( tf.to_int32(
	dictionary_from_file( dictionary_path, charset )))

def model_fn( features, labels, mode ):

    # Get the appropriate tensors
    image = features
    width = tf.size( image[1] )
    
    # Pre-process the images
    proc_image = _preprocess_image( image )
    proc_image = tf.reshape( proc_image,[1,32,-1,1] ) # Make first dim batch

    
    with tf.device(FLAGS.device):
        features,sequence_length = model.convnet_layers( proc_image, width, 
                                                         mode )
        logits = model.rnn_layers( features, sequence_length,
                                   pipeline.num_classes() )
        prediction = _get_output( logits,sequence_length )

        # predictions only takes dense tensors
        final_pred = tf.sparse_to_dense( prediction[0].indices, 
                                         prediction[0].dense_shape, 
                                         prediction[0].values, default_value=0 ) 

        return tf.estimator.EstimatorSpec( mode=mode,predictions=( final_pred ))


def main(argv=None):

    custom_config = tf.estimator.RunConfig( session_config=_get_session_config() )

    classifier = tf.estimator.Estimator( model_fn=model_fn, 
                                         model_dir=FLAGS.model,
                                         config=custom_config )

    predictions = classifier.predict( input_fn=lambda: _get_input() )

    # Get all the predictions in string format
    while True:
        try:
            print( _get_string( next( predictions )))
        except:
            sys.exit()
    
if __name__ == '__main__':
    tf.app.run()
