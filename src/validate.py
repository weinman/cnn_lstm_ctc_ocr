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

# validate.py - Run model directly on from paths to image filenames. 
# NOTE: assumes mjsynth files are given by adding an extra row of padding

import sys
import numpy as np
from PIL import Image
import tensorflow as tf

import model_fn
import charset
from lexicon import dictionary_from_file

import mjsynth

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )
tf.app.flags.DEFINE_boolean( 'print_score', False,
                             """Print log probability scores with predictions""" )
tf.app.flags.DEFINE_string( 'lexicon','',
			    """File containing lexicon of image words""" )
tf.app.flags.DEFINE_float( 'lexicon_prior',None,
			    """Prior bias [0,1] for lexicon word""" )


tf.logging.set_verbosity( tf.logging.INFO )


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

    # mjsynth input images need preprocessing transformation (shape, range)
    dataset = dataset.map( mjsynth.preprocess_image )

    # pack results for model_fn.predict 
    dataset = dataset.map ( _image_pack )
    return dataset

def _image_pack( image ):
    """
    Pack the image in a dataset into the model_fn-appropriate dictionary with 
    features and labels, where features is a dictionary containing  'image' and 'width' values.
"""
    width = tf.size( image[1] )
    
    # Pre-process the images
    proc_image = tf.reshape( image,[1,32,-1,1] ) # Make first dim batch

    # Pack the modified image data into a dictionary
    features = {'image': proc_image, 'width': width}

    # Labels unused for prediction-only validation; construct a NOP value instead
    label = tf.constant(0)
    
    return features, label


def _get_config():
    """Setup session config to soften device placement"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    custom_config = tf.estimator.RunConfig( session_config=device_config ) 

    return custom_config


def main(argv=None):
    
    classifier = tf.estimator.Estimator( config=_get_config(),
                                         model_fn=model_fn.predict_fn(
                                             FLAGS.lexicon,
                                             FLAGS.lexicon_prior), 
                                         model_dir=FLAGS.model )
    
    predictions = classifier.predict( input_fn=_get_input )
    
    # Get all the predictions in string format
    while True:
        try:
            results = next( predictions )
            print 'results =',results
            pred_str = charset.label_to_string( results['labels'] )
            if FLAGS.print_score:
                print pred_str, results['score'][0]
            else:
                print pred_str
        except StopIteration:
            sys.exit()
    
if __name__ == '__main__':
    tf.app.run()
