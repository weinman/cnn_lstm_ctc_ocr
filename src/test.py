# CNN-LSTM-CTC-OCR
# Copyright (C) 2017,2018 Jerod Weinman, Abyaya Lamsal, Benjamin Gafford
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
import time
import tensorflow as tf
from tensorflow.contrib import learn

import pipeline
import model
import filters
import model_fn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string( 'model','../data/model',
                            """Directory for model checkpoints""" )

tf.app.flags.DEFINE_integer( 'batch_size',2**9,
                             """Eval batch size""" )
tf.app.flags.DEFINE_string( 'device','/gpu:0',
                            """Device for graph placement""" )
tf.app.flags.DEFINE_string( 'test_path','../data/',
                            """Base directory for test/validation data""" )
tf.app.flags.DEFINE_string( 'filename_pattern','val/words-*',
                            """File pattern for test input data""" )
tf.app.flags.DEFINE_integer( 'num_input_threads',4,
                             """Number of readers for input data""" )
tf.app.flags.DEFINE_boolean( 'static_data', True,
                            """Whether to use static data 
                            (false for dynamic data)""" )

#tf.logging.set_verbosity( tf.logging.WARN )
#tf.logging.set_verbosity( tf.logging.INFO )

def _get_input_stream():
    if( FLAGS.static_data ):
        ds = pipeline.get_static_data( FLAGS.test_path, 
                                       str.split(
                                           FLAGS.filename_pattern,','),
                                       num_threads=FLAGS.num_input_threads,
                                       boundaries=None, # No bucketing
                                       batch_size=FLAGS.batch_size,
                                       input_device=FLAGS.device,
                                       filter_fn=None )
                                    
    else:
        ds = pipeline.get_dynamic_data( num_threads=FLAGS.num_input_threads,
                                        batch_size=FLAGS.batch_size,
                                        boundaries=None, # No bucketing
                                        input_device=FLAGS.device,
                                        filter_fn=filters.dyn_filter_by_width )

    iterator = ds.make_one_shot_iterator() 
    
    image, width, label, length, text, filename = iterator.get_next()

    # The input for the model function 
    features = {"image": image, 
                "width": width, 
                "length": length, 
                "label": label,
                "text": text,
                "filename": filename,
                "continuous_eval": False}

    return features, label

def _get_config():
    """Setup config to soften device placement and set chkpt saving intervals"""

    device_config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False )

    custom_config = tf.estimator.RunConfig(session_config=device_config)

    return custom_config


def main(argv=None):
  

    # Initialize the classifier
    classifier = tf.estimator.Estimator( config = _get_config(),
                                         model_fn=model_fn.evaluate_fn(
                                             FLAGS.device ), 
                                         model_dir=FLAGS.model)

    evaluations = classifier.evaluate( input_fn=_get_input_stream)
    print(evaluations)

if __name__ == '__main__':
    tf.app.run()
