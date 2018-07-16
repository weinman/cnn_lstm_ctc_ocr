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
from tensorflow.contrib import learn

import pipeline
import charset
import model
import model_fn
import filters

FLAGS = tf.app.flags.FLAGS

# For displaying various statistics while training
tf.logging.set_verbosity( tf.logging.INFO )

def _input_fn():
    """
    Get dataset according to tf flags for training using Estimator
    Note: Default behavior is bucketing according to default bucket boundaries
    listed in pipeline.get_data
    Returns:
      dataset : elements structured as [features, labels]
                feature structure can be seen in postbatch_fn 
                in mjsynth.py or maptextsynth.py for static or dynamic
                data pipelines respectively
    """

    # We only want a filter_fn if we have dynamic data (for now)
    filter_fn = None if FLAGS.static_data else filters.dyn_filter_by_width

    # Get data according to flags
    dataset = pipeline.get_data( FLAGS.static_data,
                                 base_dir=FLAGS.train_path,
                                 file_patterns=str.split(
                                     FLAGS.filename_pattern_train,
                                     ','),
                                 num_threads=FLAGS.num_input_threads_train,
                                 batch_size=FLAGS.batch_size_train,
                                 input_device=FLAGS.input_device,
                                 filter_fn=filter_fn )
    return dataset

def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto( allow_soft_placement=True, 
                           log_device_placement=False )

    return config 

def main( argv=None ):

    custom_config = tf.estimator.RunConfig( session_config=_get_session_config(),
                                            save_checkpoints_secs=30 )
    
    # Initialize the classifier
    classifier = tf.estimator.Estimator( model_fn=model_fn.model_fn, 
                                         model_dir=FLAGS.train_output,
                                         config=custom_config )

    # Train the model
    classifier.train( input_fn=_input_fn )

if __name__ == '__main__':
    tf.app.run()
