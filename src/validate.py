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

import os
import sys

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn

import pipeline
import model
import model_fn
from lexicon import dictionary_from_file

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )
tf.app.flags.DEFINE_string( 'device','/gpu:0',
                            """Device for graph placement""" )
tf.app.flags.DEFINE_string( 'lexicon','',
			    """File containing lexicon of image words""" )

#tf.logging.set_verbosity( tf.logging.WARN )
#tf.logging.set_verbosity( tf.logging.INFO )


def _get_image( filename ):
    """Load image data for placement in graph"""

    image = Image.open( filename ) 
    image = np.array( image )
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    image = image[:,:,:1] # so just extract first channel, preserving 3D shape

    return image


def _get_input():
    """Create a dataset of images by reading from stdin"""

    # Eliminate any trailing newline from filename
    image_data = _get_image( raw_input().rstrip() )

    # Initializing the dataset with one image
    dataset = tf.data.Dataset.from_tensors( image_data )

    # Add the rest of the images to the dataset (if any)
    for line in sys.stdin:
        image_data = _get_image( line.rstrip() )
        temp_dataset = tf.data.Dataset.from_tensors( image_data )
        dataset = dataset.concatenate( temp_dataset )
    
    # Iterate over the dataset to extract each image
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    return image


def _get_config():
    """Setup session config to soften device placement"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    custom_config = tf.estimator.RunConfig( session_config=device_config ) 

    return custom_config


def _get_string( labels ):
    """Transform an 1D array of labels into the corresponding character string"""

    string = ''.join( [pipeline.out_charset[c] for c in labels] )
    return string


def main(argv=None):
    
    classifier = tf.estimator.Estimator( config=_get_config(),
                                         model_fn=model_fn.predict_fn(
                                             FLAGS.device, FLAGS.lexicon), 
                                         model_dir=FLAGS.model )
    
    predictions = classifier.predict( input_fn=_get_input )
    
    # Get all the predictions in string format
    while True:
        try:
            print( _get_string( next( predictions )))
        except:
            sys.exit()
    
if __name__ == '__main__':
    tf.app.run()
