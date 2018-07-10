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
import time
import tensorflow as tf
from tensorflow.contrib import learn

import pipeline
import model
import filters
import model_fn

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.WARN)

def _input_fn():
    # Get data according to flags
    dataset = pipeline.get_data(FLAGS.static_data,
                           base_dir=FLAGS.eval_path,
                           file_patterns=str.split(FLAGS.filename_pattern_eval,
                                                   ','),
                           num_threads=FLAGS.num_input_threads_eval,
                           batch_size=FLAGS.batch_size_eval,
                           input_device=FLAGS.input_device,
                           filter_fn=None)
    return dataset

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def main(argv=None):
    custom_config = tf.estimator.RunConfig(session_config=_get_session_config())

    # Initialize the classifier
    classifier = tf.estimator.Estimator(model_fn=model_fn.model_fn, 
                                        model_dir=FLAGS.model,
                                        config=custom_config)
    
    while True:
        evaluations = classifier.evaluate(input_fn=_input_fn)
        print(evaluations)

if __name__ == '__main__':
    tf.app.run()
