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

optimizer = 'Adam'
tf.logging.set_verbosity(tf.logging.WARN)


def _get_input_stream():
    if(FLAGS.static_data):
        ds = pipeline.get_static_data(FLAGS.test_path, 
                                      str.split(
                                          FLAGS.filename_pattern_test,','),
                                      num_threads=FLAGS.num_input_threads_eval,
                                      boundaries=None, # No bucketing
                                      batch_size=FLAGS.batch_size_eval,
                                      input_device=FLAGS.device,
                                      filter_fn=None)
                                    
    else:
        ds = pipeline.get_dynamic_data(num_threads=FLAGS.num_input_threads_eval,
                                       batch_size=FLAGS.batch_size_eval,
                                       boundaries=None, # No bucketing
                                       input_device=FLAGS.device,
                                       filter_fn=filters.dyn_filter_by_width)

    iterator = ds.make_one_shot_iterator() 
    
    image, width, label, length, _, _ = iterator.get_next()

    # The input for the model function 
    features = {"image": image, 
                "width": width, 
                "length": length, 
                "label": label, 
                "optimizer": optimizer}

    return features, label

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
        evaluations = classifier.evaluate(input_fn=lambda: _get_input_stream())
        print(evaluations)

if __name__ == '__main__':
    tf.app.run()
